"""
검증 결과 집계. v1_parts/*.csv, eval_episodes.csv를 읽어서
표(마크다운)와 그림, 그리고 항목별 판정을 만든다.

    python verify/summarize.py
결과: verify_out/results/summary.md, summary_eval.png

집계 방식
  - 에피소드 → seed별 평균 → seed 평균들의 평균 ± 2σ (논문 Table 1과 같은 형식)
  - DAE와 CQL의 차이는 같은 seed·같은 에피소드끼리 짝지어 뺀 값의 평균 ± 2×표준오차
    (짝지은 비교라서 환경 운의 영향이 상쇄된다)
"""

import math
import os
from collections import OrderedDict, defaultdict

import numpy as np

import common as C

PRESET_ORDER = ["cql", "dae_orig", "dae_fix"]
PRESET_LABEL = {"cql": "CQL", "dae_orig": "DAE+CQL (기존)", "dae_fix": "DAE+CQL (수정)"}
COND_ORDER = ["clean", "noise", "noise_orig", "denoise", "lanereduction", "cutin"]
COND_LABEL = {"clean": "정상", "noise": "노이즈(특징별)", "noise_orig": "노이즈(기존 방식)",
              "denoise": "노이즈+DAE 복원", "lanereduction": "lane reduction", "cutin": "cut-in"}

EVAL_METRICS = OrderedDict([
    ("correction_reward", "correction_reward"),
    ("early_termination", "조기 종료 비율"),
    ("crash_by_reward", "사고 페널티 발생 비율"),
    ("visited_recon_mean", "방문 state 재구성오차"),
    ("visited_frac_above_p95", "방문 state 중 정상 p95 초과 비율"),
])


def f(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def mean_2sd(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    a = np.asarray(vals, dtype=np.float64)
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean()), float(2 * sd), len(a)


def fmt(m, s, digits=2):
    if math.isnan(m):
        return "—"
    if abs(m) < 0.01 and m != 0:
        return "{:.4f} ± {:.4f}".format(m, s)
    return "{:.{d}f} ± {:.{d}f}".format(m, s, d=digits)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ 검증 1
def section_v1(lines):
    ood = C.read_csv_parts(os.path.join(C.RESULT_DIR, "v1_parts", "ood_*.csv"))
    pen = C.read_csv_parts(os.path.join(C.RESULT_DIR, "v1_parts", "penalty_*.csv"))
    verdicts = []
    if not ood:
        lines.append("## 검증 1: DAE의 OOD 감지\n\n(v1_parts/ 결과 없음 — v1_dae_ood.py를 먼저 실행)\n")
        return verdicts

    lines.append("## 검증 1: DAE가 OOD 상태를 구분하는가\n")
    lines.append("AUROC는 0.5면 정상과 구분 못 함, 1.0이면 완전히 구분. seed가 여럿이면 평균 ± 2σ.\n")
    by_ds = defaultdict(lambda: defaultdict(list))
    kinds = {}
    for r in ood:
        by_ds[r["train_dataset"]][r["test_set"]].append((f(r["auroc_vs_clean"]), f(r["mean"])))
        kinds[r["test_set"]] = r["kind"]
    for ds in sorted(by_ds):
        rows = []
        for ts, vals in by_ds[ds].items():
            am, asd, n = mean_2sd([v[0] for v in vals])
            em, esd, _ = mean_2sd([v[1] for v in vals])
            rows.append([ts, kinds[ts], fmt(em, esd, 5), "{:.3f}".format(am) + (" ± {:.3f}".format(asd) if n > 1 else "")])
        lines.append("**{}** 로 학습한 DAE\n".format(ds))
        lines.append(md_table(["비교 대상", "종류", "평균 재구성오차", "AUROC"], rows) + "\n")

        noise02 = [v[0] for k, vv in by_ds[ds].items() if k.startswith("gaussian") and "0.2" in k for v in vv]
        other = [v[0] for k, vv in by_ds[ds].items() if kinds[k] == "other-scenario" for v in vv]
        if noise02:
            a = float(np.mean(noise02))
            verdicts.append("검증 1 ({}) 노이즈 σ=0.2 감지: AUROC {:.2f} → {}".format(
                ds, a, "구분함" if a >= 0.8 else ("부분적으로 구분" if a >= 0.65 else "거의 구분 못 함")))
        if other:
            a = float(np.mean(other))
            verdicts.append("검증 1 ({}) 다른 시나리오 감지: AUROC {:.2f} → {}".format(
                ds, a, "구분함" if a >= 0.8 else ("부분적으로 구분" if a >= 0.65 else "거의 구분 못 함")))

    if pen:
        lines.append("### 1b. 학습 중 패널티가 의미 있는 신호였나\n")
        lines.append("같은 state에 노이즈를 두 번 넣었을 때 패널티의 상관이 낮으면, 패널티는 state가 아니라 "
                     "그때 뽑힌 노이즈에 좌우된 것이다.\n")
        lam_cols = sorted({k for r in pen for k in r if k.endswith("_active_penalty_over_reward_std")})
        headers = ["데이터셋", "방식", "패널티 적용 비율", "반복 상관", "원래 오차와 순위상관"] + \
                  ["λ={} 적용 시 / reward σ".format(k.split("_")[0][3:]) for k in lam_cols]
        rows = []
        agg = defaultdict(list)
        for r in pen:
            agg[(r["train_dataset"], r["shaping"])].append(r)
        for (ds, sh), rs in sorted(agg.items()):
            row = [ds, sh,
                   "{:.1%}".format(np.mean([f(x["active_frac"]) for x in rs])),
                   "{:.2f}".format(np.mean([f(x["repeat_corr"]) for x in rs])),
                   "{:.2f}".format(np.mean([f(x["corr_with_clean_error"]) for x in rs]))]
            for k in lam_cols:
                row.append("{:.2f}배".format(np.mean([f(x[k]) for x in rs])))
            rows.append(row)
            if f(rs[0]["shape_noise_std"]) > 0:
                rc = float(np.mean([f(x["repeat_corr"]) for x in rs]))
                verdicts.append("검증 1b ({}) 기존 패널티의 반복 상관 {:.2f} → {}".format(
                    ds, rc, "state에 따라 일관됨" if rc >= 0.7 else
                    ("노이즈 영향이 큼" if rc >= 0.4 else "사실상 무작위 패널티")))
        lines.append(md_table(headers, rows) + "\n")
    return verdicts


# ------------------------------------------------------------------ 검증 2·3
def load_eval():
    rows = C.read_csv(os.path.join(C.RESULT_DIR, "eval_episodes.csv"))
    for r in rows:
        for k in EVAL_METRICS:
            r[k] = f(r.get(k))
        r["train_seed"] = int(float(r["train_seed"]))
        r["episode"] = int(float(r["episode"]))
    return rows


def per_seed(rows, ds, cond, preset, metric):
    by = defaultdict(list)
    for r in rows:
        if r["dataset"] == ds and r["condition"] == cond and r["preset"] == preset:
            by[r["train_seed"]].append(r[metric])
    return {s: float(np.nanmean(v)) for s, v in by.items() if v}


def paired_diff(rows, ds, cond, preset, metric, ref="cql"):
    """같은 (seed, episode)끼리 preset - ref."""
    a = {(r["train_seed"], r["episode"]): r[metric] for r in rows
         if r["dataset"] == ds and r["condition"] == cond and r["preset"] == preset}
    b = {(r["train_seed"], r["episode"]): r[metric] for r in rows
         if r["dataset"] == ds and r["condition"] == cond and r["preset"] == ref}
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys if not (math.isnan(a[k]) or math.isnan(b[k]))]
    if not d:
        return float("nan"), float("nan"), 0
    arr = np.asarray(d)
    se = arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else float("nan")
    return float(arr.mean()), float(2 * se), len(arr)


def section_eval(lines, rows):
    verdicts = []
    if not rows:
        lines.append("## 검증 2·3\n\n(eval_episodes.csv 없음 — evaluate.py를 먼저 실행)\n")
        return verdicts
    datasets = sorted({r["dataset"] for r in rows})
    conds = [c for c in COND_ORDER if any(r["condition"] == c for r in rows)]
    presets = [p for p in PRESET_ORDER if any(r["preset"] == p for r in rows)]

    for ds in datasets:
        lines.append("## {} 정책 평가\n".format(ds))
        for metric, label in EVAL_METRICS.items():
            head = ["조건"] + [PRESET_LABEL.get(p, p) for p in presets]
            body = []
            for c in conds:
                row = [COND_LABEL.get(c, c)]
                for p in presets:
                    ps = per_seed(rows, ds, c, p, metric)
                    m, s, n = mean_2sd(list(ps.values()))
                    row.append(fmt(m, s, 3 if "frac" in metric or "termination" in metric or "crash" in metric else 2)
                               + (" (n={})".format(n) if n else ""))
                body.append(row)
            lines.append("**{}** (seed 평균 ± 2σ)\n".format(label))
            lines.append(md_table(head, body) + "\n")

        # 짝지은 차이
        lines.append("**CQL 대비 차이** (같은 seed·에피소드끼리 뺀 값의 평균 ± 2×표준오차, 0을 벗어나면 의미 있는 차이)\n")
        head = ["조건", "비교", "correction_reward", "조기 종료 비율", "방문 state 재구성오차"]
        body = []
        for c in conds:
            for p in presets:
                if p == "cql":
                    continue
                cells = [COND_LABEL.get(c, c), PRESET_LABEL.get(p, p)]
                for metric in ("correction_reward", "early_termination", "visited_recon_mean"):
                    m, s, n = paired_diff(rows, ds, c, p, metric)
                    if n == 0:
                        cells.append("—")
                        continue
                    sig = (not math.isnan(s)) and (m - s > 0 or m + s < 0)
                    cells.append(("{:+.4f} ± {:.4f}" if abs(m) < 0.01 else "{:+.2f} ± {:.2f}").format(m, s)
                                 + (" *" if sig else ""))
                body.append(cells)
        lines.append(md_table(head, body) + "\n")

        # 판정
        for p in presets:
            if p == "cql":
                continue
            # 검증 2: 방문 state 오차가 낮아졌는가
            sig_lower = []
            for c in conds:
                m, s, n = paired_diff(rows, ds, c, p, "visited_recon_mean")
                if n and not math.isnan(s):
                    sig_lower.append((c, m + s < 0, m - s > 0))
            lower = [c for c, lo, hi in sig_lower if lo]
            higher = [c for c, lo, hi in sig_lower if hi]
            v2 = ("지지 (방문 state 오차가 유의하게 낮음: {})".format(", ".join(COND_LABEL[c] for c in lower))
                  if lower and not higher else
                  "반증 (오히려 높음: {})".format(", ".join(COND_LABEL[c] for c in higher)) if higher and not lower else
                  "판단 보류 (유의한 차이 없음 또는 조건마다 엇갈림)")
            verdicts.append("검증 2 ({}, {}) 익숙한 상태에 머무는가: {}".format(ds, PRESET_LABEL[p], v2))

            # 검증 3: 낯선 조건(노이즈·다른 시나리오)에서 더 안전/더 좋은가
            hard = [c for c in conds if c in ("noise", "noise_orig", "lanereduction", "cutin")]
            better, worse = [], []
            for c in hard:
                m, s, n = paired_diff(rows, ds, c, p, "correction_reward")
                mt, st, _ = paired_diff(rows, ds, c, p, "early_termination")
                if n and not math.isnan(s):
                    if m - s > 0 or (not math.isnan(st) and mt + st < 0):
                        better.append(c)
                    if m + s < 0 or (not math.isnan(st) and mt - st > 0):
                        worse.append(c)
            v3 = ("지지 ({}에서 CQL보다 나음)".format(", ".join(COND_LABEL[c] for c in better))
                  if better and not worse else
                  "반증 ({}에서 CQL보다 나쁨)".format(", ".join(COND_LABEL[c] for c in worse)) if worse and not better else
                  "엇갈림 (나음: {} / 나쁨: {})".format(", ".join(COND_LABEL[c] for c in better) or "없음",
                                                    ", ".join(COND_LABEL[c] for c in worse) or "없음")
                  if better and worse else "판단 보류 (유의한 차이 없음)")
            verdicts.append("검증 3 ({}, {}) 낯선 조건에서 더 안전한가: {}".format(ds, PRESET_LABEL[p], v3))
    return verdicts


def plot_eval(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[summary] 그림 건너뜀: {}".format(exc))
        return
    if not rows:
        return
    datasets = sorted({r["dataset"] for r in rows})
    conds = [c for c in COND_ORDER if any(r["condition"] == c for r in rows)]
    presets = [p for p in PRESET_ORDER if any(r["preset"] == p for r in rows)]
    colors = {"cql": "#8a8f98", "dae_orig": "#c9a26b", "dae_fix": "#2f6fb0"}
    names = {"cql": "CQL", "dae_orig": "DAE+CQL (original)", "dae_fix": "DAE+CQL (fixed)"}
    cond_en = {"clean": "clean", "noise": "noise", "noise_orig": "noise (orig)", "denoise": "noise+denoise",
               "lanereduction": "lane reduction", "cutin": "cut-in"}
    metrics = [("correction_reward", "correction reward"),
               ("early_termination", "early termination rate"),
               ("visited_recon_mean", "visited-state recon. error")]
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(4.6 * len(metrics), 3.4 * len(datasets) + 0.4),
                             squeeze=False)
    handles = []
    for i, ds in enumerate(datasets):
        for j, (m, lab) in enumerate(metrics):
            ax = axes[i][j]
            is_rate = m in ("early_termination", "crash_by_reward") or m.startswith("visited_frac")
            w = 0.8 / max(1, len(presets))
            for k, p in enumerate(presets):
                xs, ys, lo, hi = [], [], [], []
                for ci, c in enumerate(conds):
                    ps = per_seed(rows, ds, c, p, m)
                    mu, sd2, n = mean_2sd(list(ps.values()))
                    if n:
                        xs.append(ci + (k - (len(presets) - 1) / 2.0) * w)
                        ys.append(mu)
                        low, high = mu - sd2, mu + sd2
                        if is_rate:  # 비율은 0~1 밖으로 나갈 수 없다
                            low, high = max(0.0, low), min(1.0, high)
                        lo.append(mu - low)
                        hi.append(high - mu)
                h = ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=colors.get(p, "#333"), capsize=2,
                                markersize=4, linewidth=1, label=names.get(p, p))
                if i == 0 and j == 0:
                    handles.append(h)
            if is_rate:
                ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels([cond_en[c] for c in conds], rotation=30, ha="right", fontsize=8)
            ax.set_title("{} — {}".format(ds, lab), fontsize=9)
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, frameon=False)
    fig.text(0.01, 0.005, "points: mean over seeds, bars: ±2σ across seeds (rates clipped to [0, 1])",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(path, dpi=150)
    print("[summary] 그림 저장: {}".format(path))


def main():
    C.ensure_dirs()
    lines = ["# DAE+CQL 의도 검증 결과\n"]
    verdicts = section_v1(lines)
    rows = load_eval()
    verdicts += section_eval(lines, rows)
    plot_eval(rows, os.path.join(C.RESULT_DIR, "summary_eval.png"))

    lines.insert(1, "## 판정 요약\n\n" + ("\n".join("- " + v for v in verdicts) if verdicts else "- (결과 없음)") + "\n")
    out = os.path.join(C.RESULT_DIR, "summary.md")
    with open(out, "w") as fp:
        fp.write("\n".join(lines))
    print("\n".join(lines[:2]))
    print("[summary] 저장: {}".format(out))


if __name__ == "__main__":
    main()

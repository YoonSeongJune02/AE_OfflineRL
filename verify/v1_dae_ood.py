"""
검증 1: DAE가 OOD 상태를 구분하는가, 그리고 학습 중 패널티가 의미 있는 신호인가.

실행 예:
    python verify/v1_dae_ood.py --dataset highway-NGSIM --seed 5

하는 일
  1a. DAE를 학습 데이터 90%로 학습하고 10%는 끝까지 안 보이게 둔다(진짜 held-out).
      held-out 정상 state와 여러 OOD state의 재구성오차를 비교하고 AUROC를 낸다.
        - 가우시안 노이즈 (0.1, 0.2, 0.5), 마스킹 노이즈
        - 같은 시나리오의 다른 데이터셋 (예: highway-random)
        - 다른 시나리오 데이터 (lanereduction-*, cutin-*)
  1b. 기존 학습 방식 그대로 패널티를 흉내 내서
        - 같은 state에 노이즈를 두 번 넣었을 때 패널티가 비슷하게 나오는지 (일관성)
        - 패널티가 실제로 얼마나 자주, 얼마나 크게 걸리는지 (reward 표준편차 대비)
      를 잰다. 기존 방식(노이즈 0.2 후 계산)과 노이즈 없는 방식을 같이 본다.

산출물 (verify_out/results/)
  v1_parts/ood_<dataset>_seed<seed>.csv, v1_parts/penalty_<dataset>_seed<seed>.csv,
  v1_hist_<dataset>_seed<seed>.png  (summarize.py가 합친다)
  DAE 체크포인트: verify_out/checkpoints/dae/<dataset>_seed<seed>.pt  (검증 2·3에서 재사용)
"""

import argparse
import os
import time

import numpy as np
import torch

import common as C
from Algos.DAE_v2 import pretrain_dae
from Algos.reward_shaping_v2 import RewardShaper


def parse_args():
    p = argparse.ArgumentParser(description="검증 1: DAE OOD 감지와 패널티 신호")
    p.add_argument("--dataset", required=True, help="DAE를 학습할 데이터셋 (예: highway-NGSIM)")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.1, help="끝까지 안 보이게 둘 held-out 비율")
    p.add_argument("--retrain", action="store_true", help="체크포인트가 있어도 DAE를 다시 학습")
    p.add_argument("--dae-epochs", type=int, default=C.DEFAULT_DAE_CFG["epochs"])
    p.add_argument("--n-eval", type=int, default=20000, help="조건마다 평가에 쓸 state 수")
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.1, 0.2, 0.5])
    p.add_argument("--mask-prob", type=float, default=0.2)
    p.add_argument("--ood-datasets", nargs="*", default=None,
                   help="비교할 다른 데이터셋. 비우면 buffers/ 안의 나머지 전부")
    # 1b: 패널티 흉내
    p.add_argument("--shape-noise-std", type=float, default=0.2, help="기존 학습 코드가 쓰던 값")
    p.add_argument("--threshold-z", type=float, default=1.0)
    p.add_argument("--shaper-steps", type=int, default=3000)
    p.add_argument("--shaper-batch", type=int, default=64)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    return p.parse_args()


def summarize(errors, clean_errors, thr_p95):
    return {
        "n": len(errors),
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "p50": float(np.percentile(errors, 50)),
        "p95": float(np.percentile(errors, 95)),
        "auroc_vs_clean": C.auroc(clean_errors, errors),
        "frac_above_clean_p95": float(np.mean(errors > thr_p95)),
    }


def train_or_load_dae(args, states_train, device):
    path = C.dae_ckpt_path(args.dataset, args.seed)
    if os.path.isfile(path) and not args.retrain:
        dae, cfg, meta = C.load_dae_ckpt(path, device)
        print("[v1] 저장된 DAE 사용: {}".format(path))
        return dae, cfg, meta, path

    cfg = dict(C.DEFAULT_DAE_CFG)
    cfg["epochs"] = args.dae_epochs
    dae = C.build_dae(cfg, device)
    buf = C.StateBuffer(states_train, device)
    t0 = time.time()
    dae, hist = pretrain_dae(
        dae, buf, device,
        noise_type=cfg["noise_type"], noise_std=cfg["noise_std"], mask_prob=cfg["mask_prob"],
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], val_split=0.1, patience=cfg["patience"],
        log_wandb=False,
    )
    meta = {"best_epoch": hist["best_epoch"], "train_minutes": (time.time() - t0) / 60.0}
    return dae, cfg, meta, path


def simulate_shaper(dae, states, rewards, device, noise_std, threshold_z, steps, batch, rng):
    """기존 학습 루프와 같은 방식으로 RewardShaper를 돌려 샘플별 패널티 p (λ 곱하기 전)를 모은다."""
    shaper = RewardShaper(lambda_max=1.0, warmup_steps=0, threshold_z=threshold_z,
                          use_normalization=True, use_tanh=True)
    n = len(states)
    ps = []
    burn_in = min(200, steps // 10)
    for t in range(steps):
        ind = rng.randint(0, n, size=batch)
        s = torch.from_numpy(states[ind]).to(device)
        r = torch.from_numpy(rewards[ind].reshape(-1, 1).astype(np.float32)).to(device)
        if noise_std > 0:
            s = s + torch.randn_like(s) * noise_std
        e = dae.reconstruction_error(s)
        shaped, _ = shaper.shape(r, e)
        if t >= burn_in:
            ps.append((r - shaped).view(-1).cpu().numpy())  # λ=1 이므로 r - shaped = p
    return np.concatenate(ps).astype(np.float64), shaper.running.mean, shaper.running.std


def penalty_from_errors(e, mu, sd, thr):
    z = (e - mu) / sd
    return np.tanh(np.clip(z - thr, 0.0, None))


def main():
    args = parse_args()
    C.ensure_dirs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = C.get_device()
    rng = np.random.RandomState(args.seed)

    if C.scenario_of(args.dataset) is None:
        raise SystemExit("시나리오를 알 수 없는 데이터셋 이름: {}".format(args.dataset))

    # ---------- 데이터 ----------
    states = C.load_states(args.dataset)
    rewards = C.load_rewards(args.dataset)
    n_common = min(len(states), len(rewards))
    states, rewards = states[:n_common], rewards[:n_common]
    tr_idx, ho_idx = C.split_indices(len(states), args.val_frac, seed=args.seed)
    print("[v1] {}: 전체 {}개, 학습 {}개, held-out {}개".format(
        args.dataset, len(states), len(tr_idx), len(ho_idx)))

    # ---------- DAE ----------
    dae, cfg, meta, ckpt_path = train_or_load_dae(args, states[tr_idx], device)

    heldout = states[ho_idx]
    if len(heldout) > args.n_eval:
        heldout = heldout[rng.choice(len(heldout), args.n_eval, replace=False)]
    clean_err = C.recon_error_np(dae, heldout, device)
    thr_p95 = float(np.percentile(clean_err, 95))
    thr_p99 = float(np.percentile(clean_err, 99))

    meta.update({
        "dataset": args.dataset, "seed": args.seed, "val_frac": args.val_frac,
        "clean_err_mean": float(clean_err.mean()), "clean_err_std": float(clean_err.std()),
        "thr_p95": thr_p95, "thr_p99": thr_p99,
        "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()),
    })
    C.save_dae_ckpt(ckpt_path, dae, cfg, meta)
    print("[v1] DAE 저장: {}".format(ckpt_path))

    # ---------- 1a. OOD 비교 ----------
    rows = []
    base = {"train_dataset": args.dataset, "seed": args.seed}
    rows.append(dict(base, test_set="held-out (clean)", kind="in-dist",
                     **summarize(clean_err, clean_err, thr_p95)))

    for sd in args.noise_levels:
        noisy = heldout + rng.randn(*heldout.shape).astype(np.float32) * sd
        e = C.recon_error_np(dae, noisy, device)
        rows.append(dict(base, test_set="gaussian σ={}".format(sd), kind="noise",
                         **summarize(e, clean_err, thr_p95)))

    mask = (rng.rand(*heldout.shape) > args.mask_prob).astype(np.float32)
    e = C.recon_error_np(dae, heldout * mask, device)
    rows.append(dict(base, test_set="masking p={}".format(args.mask_prob), kind="noise",
                     **summarize(e, clean_err, thr_p95)))

    if args.ood_datasets is None:
        others = sorted(d for d in os.listdir(C.BUFFER_DIR)
                        if d != args.dataset and C.buffer_exists(d) and C.scenario_of(d))
    else:
        others = args.ood_datasets
    my_scn = C.scenario_of(args.dataset)
    hist_examples = {"held-out (clean)": clean_err}
    for d in others:
        if not C.buffer_exists(d):
            print("[v1] 건너뜀 (버퍼 없음): {}".format(d))
            continue
        s = C.load_states(d, max_n=args.n_eval, seed=args.seed)
        e = C.recon_error_np(dae, s, device)
        kind = "other-dataset" if C.scenario_of(d) == my_scn else "other-scenario"
        rows.append(dict(base, test_set=d, kind=kind, **summarize(e, clean_err, thr_p95)))
        if kind == "other-scenario" and "other scenario: " + d.split("-")[0] not in hist_examples:
            hist_examples["other scenario: " + d.split("-")[0]] = e

    noisy02 = heldout + rng.randn(*heldout.shape).astype(np.float32) * 0.2
    hist_examples["gaussian σ=0.2"] = C.recon_error_np(dae, noisy02, device)

    fields_ood = ["train_dataset", "seed", "test_set", "kind", "n", "mean", "std", "p50", "p95",
                  "auroc_vs_clean", "frac_above_clean_p95"]
    ood_path = os.path.join(C.RESULT_DIR, "v1_parts", "ood_{}_seed{}.csv".format(args.dataset, args.seed))
    if os.path.isfile(ood_path):
        os.remove(ood_path)  # 다시 돌려도 중복되지 않게 실행마다 새로 쓴다
    C.append_csv(ood_path, rows, fields_ood)

    print("\n[1a] 재구성오차 (AUROC 0.5 = 구분 못 함, 1.0 = 완전 구분)")
    print("{:<34} {:>15} {:>10} {:>8}".format("비교 대상", "종류", "평균 오차", "AUROC"))
    for r in rows:
        print("{:<34} {:>15} {:>10.5f} {:>8.3f}".format(r["test_set"], r["kind"], r["mean"], r["auroc_vs_clean"]))

    # ---------- 1b. 학습 중 패널티 신호 ----------
    reward_std = float(rewards.std())
    pen_rows = []
    probe = states[rng.choice(len(states), min(args.n_eval, len(states)), replace=False)]
    e_clean = C.recon_error_np(dae, probe, device)

    for label, sd in [("기존 방식 (노이즈 {} 후 계산)".format(args.shape_noise_std), args.shape_noise_std),
                      ("노이즈 없이 계산", 0.0)]:
        p, mu, sdv = simulate_shaper(dae, states, rewards, device, sd, args.threshold_z,
                                     args.shaper_steps, args.shaper_batch, rng)
        # 같은 state에 노이즈를 두 번 넣었을 때 패널티가 같게 나오는가
        if sd > 0:
            e1 = C.recon_error_np(dae, probe + rng.randn(*probe.shape).astype(np.float32) * sd, device)
            e2 = C.recon_error_np(dae, probe + rng.randn(*probe.shape).astype(np.float32) * sd, device)
            p1 = penalty_from_errors(e1, mu, sdv, args.threshold_z)
            p2 = penalty_from_errors(e2, mu, sdv, args.threshold_z)
            rep = C.pearson(p1, p2)
            active_either = np.logical_or(p1 > 0, p2 > 0)
            agree = float(np.mean((p1 > 0) == (p2 > 0))) if active_either.any() else float("nan")
            jacc = float(np.logical_and(p1 > 0, p2 > 0).sum() / max(1, active_either.sum()))
            link = C.spearman(e_clean, e1)
        else:
            rep, agree, jacc, link = 1.0, 1.0, 1.0, 1.0

        row = {
            "train_dataset": args.dataset, "seed": args.seed, "shaping": label,
            "shape_noise_std": sd, "threshold_z": args.threshold_z,
            "active_frac": float(np.mean(p > 0)),
            "p_mean": float(p.mean()),
            "p_mean_when_active": float(p[p > 0].mean()) if (p > 0).any() else 0.0,
            "p_p95": float(np.percentile(p, 95)),
            "repeat_corr": rep, "repeat_active_jaccard": jacc, "repeat_active_agree": agree,
            "corr_with_clean_error": link,
            "reward_std": reward_std,
        }
        for lam in args.lambdas:
            row["lam{}_mean_penalty_over_reward_std".format(lam)] = lam * row["p_mean"] / reward_std
            row["lam{}_active_penalty_over_reward_std".format(lam)] = lam * row["p_mean_when_active"] / reward_std
        pen_rows.append(row)

    fields_pen = ["train_dataset", "seed", "shaping", "shape_noise_std", "threshold_z",
                  "active_frac", "p_mean", "p_mean_when_active", "p_p95",
                  "repeat_corr", "repeat_active_jaccard", "repeat_active_agree",
                  "corr_with_clean_error", "reward_std"]
    for lam in args.lambdas:
        fields_pen += ["lam{}_mean_penalty_over_reward_std".format(lam),
                       "lam{}_active_penalty_over_reward_std".format(lam)]
    pen_path = os.path.join(C.RESULT_DIR, "v1_parts", "penalty_{}_seed{}.csv".format(args.dataset, args.seed))
    if os.path.isfile(pen_path):
        os.remove(pen_path)
    C.append_csv(pen_path, pen_rows, fields_pen)

    print("\n[1b] 학습 중 패널티 신호 (reward 표준편차 {:.3f})".format(reward_std))
    for r in pen_rows:
        print("  - {}".format(r["shaping"]))
        print("      패널티가 걸리는 비율 {:.1%}, 걸렸을 때 평균 p {:.3f}".format(
            r["active_frac"], r["p_mean_when_active"]))
        print("      같은 state 반복 시 패널티 상관 {:.3f}, 원래 state 오차와의 순위상관 {:.3f}".format(
            r["repeat_corr"], r["corr_with_clean_error"]))
        for lam in args.lambdas:
            print("      λ={}: 걸렸을 때 패널티 = reward 표준편차의 {:.2f}배".format(
                lam, r["lam{}_active_penalty_over_reward_std".format(lam)]))

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        all_vals = np.concatenate(list(hist_examples.values()))
        lo = max(1e-6, np.percentile(all_vals, 0.5))
        hi = np.percentile(all_vals, 99.5)
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
        greys = ["#3b6ea5", "#9aa5b1", "#c47a3b", "#6b8f71", "#8c6bb1"]
        for i, (name, vals) in enumerate(hist_examples.items()):
            ax.hist(vals, bins=bins, histtype="step", linewidth=1.6,
                    weights=np.ones(len(vals)) / max(1, len(vals)),
                    color=greys[i % len(greys)], label=name)
        ax.axvline(thr_p95, color="#555555", linestyle="--", linewidth=1)
        ax.text(thr_p95, ax.get_ylim()[1] * 0.92, " clean p95", fontsize=8, color="#555555")
        ax.set_xscale("log")
        ax.set_xlabel("reconstruction error (MSE vs input, log scale)")
        ax.set_ylabel("fraction of states")
        ax.set_title("{} DAE: clean held-out vs OOD states".format(args.dataset))
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        out = os.path.join(C.RESULT_DIR, "v1_hist_{}_seed{}.png".format(args.dataset, args.seed))
        fig.savefig(out, dpi=150)
        print("\n[v1] 그림 저장: {}".format(out))
    except Exception as exc:  # 서버에 matplotlib/한글 폰트가 없어도 결과는 CSV로 남는다
        print("[v1] 그림은 건너뜀: {}".format(exc))


if __name__ == "__main__":
    main()

"""
검증 스크립트 공통 유틸.

- 버퍼 로딩 (state / reward)
- 진짜 held-out 분리 (기존 pretrain_dae는 train/val을 같은 버퍼에서 뽑음)
- DAE 체크포인트 저장/로드
- 재구성오차 계산 (입력 기준: ||DAE(x) - x||^2 평균)
- AUROC (sklearn 없이 순위 기반으로 계산)

Python 3.7 / 구버전 torch에서도 돌아가도록 작성했다.
"""

import csv
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Algos.DAE_v2 import DAE  # noqa: E402

BUFFER_DIR = os.environ.get("AE_BUFFER_DIR", os.path.join(REPO_ROOT, "buffers"))
OUT_DIR = os.environ.get("AE_VERIFY_OUT", os.path.join(REPO_ROOT, "verify_out"))
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
RESULT_DIR = os.path.join(OUT_DIR, "results")

STATE_DIM = 19
ACTION_DIM = 2
MAX_ACTION = 1.0

HIGHWAY_DATASETS = [
    "highway-NGSIM", "highway-final", "highway-medium", "highway-random",
    "highway-final-medium", "highway-final-random", "highway-humanlike",
]

# 데이터셋 접두어 -> 시나리오 설정 이름
# MA_5LC = highway, MA_4BL = lane reduction(병목 링), UnifiedRing = cut-in
SCENARIO_OF_PREFIX = {
    "highway": "MA_5LC",
    "lanereduction": "MA_4BL",
    "cutin": "UnifiedRing",
}

WARMUP_TS = {"MA_5LC": 125, "MA_4BL": 900, "UnifiedRing": 90}


def scenario_of(dataset):
    prefix = dataset.split("-")[0].lower()
    return SCENARIO_OF_PREFIX.get(prefix)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    for d in (CKPT_DIR, RESULT_DIR):
        if not os.path.isdir(d):
            os.makedirs(d)


def buffer_exists(dataset):
    return os.path.isfile(os.path.join(BUFFER_DIR, dataset, "state.npy"))


def load_states(dataset, max_n=None, seed=0, field="state"):
    """버퍼에서 state(또는 next_state)를 float32로 읽는다. max_n이 있으면 무작위로 그만큼만."""
    path = os.path.join(BUFFER_DIR, dataset, field + ".npy")
    arr = np.load(path, mmap_mode="r", allow_pickle=True)
    n = arr.shape[0]
    if max_n is not None and n > max_n:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(n, size=max_n, replace=False))
        return np.asarray(arr[idx], dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def load_rewards(dataset):
    r = np.load(os.path.join(BUFFER_DIR, dataset, "reward.npy"), allow_pickle=True)
    return np.asarray(r, dtype=np.float64).reshape(-1)


def split_indices(n, val_frac=0.1, seed=0):
    """진짜 held-out 분리. 앞은 학습용, 뒤는 평가 전용."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    return perm[n_val:], perm[:n_val]


class StateBuffer(object):
    """pretrain_dae가 요구하는 최소 인터페이스(.size, .sample()[0])만 갖춘 버퍼."""

    def __init__(self, states, device):
        self.states = np.asarray(states, dtype=np.float32)
        self.size = self.states.shape[0]
        self.device = device

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (torch.from_numpy(self.states[ind]).to(self.device),)


DEFAULT_DAE_CFG = {
    "latent_dim": 32,
    "hidden_dim": 128,
    "dropout": 0.1,
    "noise_type": "mixed",
    "noise_std": 0.1,
    "mask_prob": 0.05,
    "epochs": 100,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 10,
}


def build_dae(cfg, device):
    return DAE(
        state_dim=STATE_DIM,
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)


def dae_ckpt_path(dataset, seed):
    return os.path.join(CKPT_DIR, "dae", "{}_seed{}.pt".format(dataset, seed))


def save_dae_ckpt(path, dae, cfg, meta):
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        os.makedirs(d)
    torch.save({"state_dict": dae.state_dict(), "cfg": cfg, "meta": meta}, path)


def load_dae_ckpt(path, device):
    ck = torch.load(path, map_location=device)
    dae = build_dae(ck["cfg"], device)
    dae.load_state_dict(ck["state_dict"])
    dae.eval()
    return dae, ck["cfg"], ck["meta"]


def recon_error_np(dae, states, device, batch=8192):
    """입력 기준 재구성오차 (샘플별 MSE). states: (N, 19) numpy."""
    out = []
    dae.eval()
    with torch.no_grad():
        for i in range(0, len(states), batch):
            x = torch.from_numpy(np.asarray(states[i:i + batch], dtype=np.float32)).to(device)
            out.append(dae.reconstruction_error(x).view(-1).cpu().numpy())
    if not out:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(out).astype(np.float64)


def denoise_np(dae, states, device):
    """DAE 출력(복원된 state)을 돌려준다. 평가 때 '입력 정화' 조건에 쓴다."""
    dae.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(device)
        if x.dim() == 1:
            x = x.view(1, -1)
            return dae(x).view(-1).cpu().numpy()
        return dae(x).cpu().numpy()


def _rankdata(x):
    """동점은 평균 순위로 처리하는 순위 (scipy.stats.rankdata average와 동일)."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def auroc(neg_scores, pos_scores):
    """pos(=OOD)가 neg(=정상)보다 점수가 클 확률. 0.5면 구분 못 함, 1.0이면 완전 구분."""
    neg = np.asarray(neg_scores, dtype=np.float64)
    pos = np.asarray(pos_scores, dtype=np.float64)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([neg, pos]))
    r_pos = ranks[len(neg):].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(_rankdata(a), _rankdata(b))


def append_csv(path, rows, fieldnames):
    """헤더가 없으면 쓰고, 행을 이어 붙인다."""
    new = not os.path.isfile(path)
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_csv_parts(pattern):
    """glob 패턴에 맞는 CSV들을 전부 읽어 합친다."""
    import glob
    rows = []
    for path in sorted(glob.glob(pattern)):
        rows.extend(read_csv(path))
    return rows

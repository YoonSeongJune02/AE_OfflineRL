"""
검증 2·3: 저장된 정책들을 같은 조건에서 평가한다.

  검증 2 (메커니즘): 주행 중 방문한 state의 재구성오차.
      DAE+CQL이 의도대로 "익숙한 상태"에 머무른다면, 같은 측정용 DAE로 쟀을 때
      방문 state의 오차가 CQL보다 낮아야 한다.
  검증 3 (목표): 같은 조건에서의 성능과 안전 지표.
      correction_reward, 조기 종료(사고) 비율, 사고 페널티 발생 여부.

조건 (--conditions)
  clean          : 학습한 시나리오, 입력 그대로
  noise          : 입력에 특징별 독립 가우시안 노이즈 (σ = --noise-std)
  noise_orig     : 기존 평가 코드와 같은 노이즈 (값 하나를 뽑아 19개 특징에 똑같이 더함)
  denoise        : noise와 같은 노이즈를 넣은 뒤 DAE로 복원해서 정책에 넣음 (입력 정화 대안)
  lanereduction  : lane reduction 시나리오(MA_4BL)에서 평가 — 학습 때 못 본 상황
  cutin          : cut-in 시나리오(UnifiedRing)에서 평가 — 학습 때 못 본 상황

공정한 비교를 위해 한 프로세스에서 여러 정책을 같은 SUMO 시드, 같은 에피소드 시드,
같은 노이즈로 평가한다(짝지은 비교). 원래 코드처럼 PPO 에이전트/ray 워커를 띄우지 않고
환경 하나만 만들어서 SUMO 인스턴스 수를 줄였다.

실행 예 (서버, ad4rl 환경, 저장소 루트에서):
    python verify/evaluate.py --dataset highway-NGSIM --seed 5 \
        --presets cql dae_orig dae_fix --conditions clean noise noise_orig denoise cutin lanereduction \
        --episodes 5
결과: verify_out/results/eval_episodes.csv (에피소드마다 한 줄)
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

import common as C

ACCIDENT_REWARD_THRESHOLD = -4.5  # 설정 파일 reward_params['accident_penalty'] = -5

CONDITION_SCENARIO = {"lanereduction": "MA_4BL", "cutin": "UnifiedRing", "highway": "MA_5LC"}

FIELDS = ["run", "preset", "dataset", "train_seed", "condition", "scenario", "noise_std",
          "episode", "eval_seed", "steps", "early_termination", "crash_by_reward", "crash_by_sim",
          "vanilla_reward", "timesteps", "correction_reward",
          "visited_recon_mean", "visited_recon_p95", "visited_frac_above_p95", "visited_frac_above_p99",
          "mean_obs0", "wall_sec"]


def parse_args():
    p = argparse.ArgumentParser(description="검증 2·3: 저장된 정책을 같은 조건에서 평가")
    p.add_argument("--dataset", required=True, help="정책을 학습한 데이터셋")
    p.add_argument("--seed", type=int, required=True, help="정책을 학습한 seed")
    p.add_argument("--presets", nargs="+", default=["cql", "dae_orig", "dae_fix"])
    p.add_argument("--tag", default="", help="train_policy.py에서 쓴 --tag")
    p.add_argument("--which", default="final", help="final 또는 ep10 같은 중간 저장")
    p.add_argument("--conditions", nargs="+",
                   default=["clean", "noise", "noise_orig", "denoise", "cutin", "lanereduction"])
    p.add_argument("--noise-std", type=float, default=0.2)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--eval-seed", type=int, default=1000,
                   help="SUMO 시드와 에피소드 시드의 기준값. 정책끼리 같은 값을 써야 짝지은 비교가 된다")
    p.add_argument("--dae-ckpt", default=None,
                   help="측정용 DAE. 기본값: verify_out/checkpoints/dae/<dataset>_seed<seed>.pt")
    p.add_argument("--max-steps", type=int, default=4000, help="에피소드 안전 상한")
    p.add_argument("--out", default=None, help="결과 CSV 경로 (기본: verify_out/results/eval_episodes.csv)")
    return p.parse_args()


# ---------------------------------------------------------------- policies
class LoadedActor(object):
    def __init__(self, path, device):
        from Algos.DDPG_CQL import Actor
        self.actor = Actor(C.STATE_DIM, C.ACTION_DIM, C.MAX_ACTION).to(device)
        self.actor.load_state_dict(torch.load(path, map_location=device))
        self.actor.eval()
        self.device = device

    def act(self, obs):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(self.device)
            return self.actor(x).cpu().numpy().flatten()


def find_runs(args):
    runs = []
    for preset in args.presets:
        name = "{}_{}_seed{}".format(preset, args.dataset, args.seed) + ("_" + args.tag if args.tag else "")
        d = os.path.join(C.CKPT_DIR, "policy", name)
        path = os.path.join(d, "actor_{}.pt".format(args.which))
        if not os.path.isfile(path):
            print("[eval] 정책 없음, 건너뜀: {}".format(path))
            continue
        runs.append((preset, name, path))
    return runs


# ---------------------------------------------------------------- environment
def make_env(exp_config):
    """원래 main의 환경 생성에서 PPO 에이전트/ray.init을 뺀 것."""
    from flow.utils.registry import make_create_env
    module_ma = __import__("exp_configs.rl.multiagent", fromlist=[exp_config])
    submodule = getattr(module_ma, exp_config)
    # 설정 파일이 import될 때 test_env를 하나 만들어 SUMO를 띄운다. 쓰지 않으니 닫는다.
    test_env = getattr(submodule, "test_env", None)
    if test_env is not None:
        try:
            test_env.terminate()
        except Exception:
            pass
    create_env, _ = make_create_env(params=submodule.flow_params, version=0)
    env = create_env()
    horizon = 3000
    try:
        horizon = int(env.env_params.horizon)
    except Exception:
        pass
    return env, horizon


def set_sumo_seed(env, seed):
    """restart_instance=True면 reset 때 SUMO를 다시 띄우므로, 그 전에 시드를 바꿔 둔다."""
    for obj in (env, getattr(env, "unwrapped", None)):
        sp = getattr(obj, "sim_params", None)
        if sp is not None:
            try:
                sp.seed = int(seed)
                return True
            except Exception:
                pass
    return False


def sim_collision(env):
    try:
        return bool(env.k.simulation.check_collision())
    except Exception:
        return None


def final_timestep(env, steps, warmup_ts):
    """원래 코드의 timesteps 계산을 그대로 따른다. 실패하면 warmup + step 수로 대신한다."""
    try:
        k = env.unwrapped.k if hasattr(env, "unwrapped") else env.k
        return float(k.vehicle.get_timestep(k.vehicle.get_ids()[1]) / 100.0)
    except Exception:
        return float(warmup_ts + steps)


# ---------------------------------------------------------------- one episode
def run_episode(env, horizon, warmup_ts, actor, cond, noise_std, dae, thr95, thr99, device, ep_seed,
                max_steps):
    rng = np.random.RandomState(ep_seed)  # 정책끼리 같은 노이즈를 받도록 에피소드마다 고정
    try:
        env.seed(ep_seed)
    except Exception:
        pass
    np.random.seed(ep_seed)
    random.seed(ep_seed)

    state = env.reset()
    agent_id = list(state.keys())[0]
    tot_reward, steps = 0.0, 0
    crash_r, crash_s = False, False
    visited = []
    t0 = time.time()

    while steps < max_steps:
        obs = np.asarray(list(state.values())[0], dtype=np.float32).reshape(-1)
        visited.append(obs)

        if cond in ("noise", "denoise"):
            x = obs + rng.randn(obs.shape[0]).astype(np.float32) * noise_std
            if cond == "denoise":
                x = C.denoise_np(dae, x, device)
        elif cond == "noise_orig":
            x = obs + np.float32(rng.randn() * noise_std)  # 기존 평가: 값 하나를 전 특징에 더함
        else:
            x = obs

        action = actor.act(x)
        state, reward, done, _ = env.step({agent_id: action})
        r = float(list(reward.values())[0]) if isinstance(reward, dict) else float(reward)
        tot_reward += r
        steps += 1
        if r <= ACCIDENT_REWARD_THRESHOLD:
            crash_r = True
        sc = sim_collision(env)
        if sc:
            crash_s = True

        is_done = done["__all__"] if isinstance(done, dict) else bool(done)
        if is_done:
            break

    ts = final_timestep(env, steps, warmup_ts)
    visited = np.stack(visited) if visited else np.zeros((0, C.STATE_DIM), dtype=np.float32)
    e = C.recon_error_np(dae, visited, device) if len(visited) else np.zeros(0)
    return {
        "steps": steps,
        "early_termination": int(steps < horizon - 1),
        "crash_by_reward": int(crash_r),
        "crash_by_sim": int(crash_s),
        "vanilla_reward": tot_reward,
        "timesteps": ts,
        "correction_reward": tot_reward + ts - warmup_ts,
        "visited_recon_mean": float(e.mean()) if len(e) else float("nan"),
        "visited_recon_p95": float(np.percentile(e, 95)) if len(e) else float("nan"),
        "visited_frac_above_p95": float(np.mean(e > thr95)) if len(e) else float("nan"),
        "visited_frac_above_p99": float(np.mean(e > thr99)) if len(e) else float("nan"),
        "mean_obs0": float(visited[:, 0].mean()) if len(visited) else float("nan"),
        "wall_sec": time.time() - t0,
    }


def main():
    args = parse_args()
    C.ensure_dirs()
    device = C.get_device()
    out_path = args.out or os.path.join(C.RESULT_DIR, "eval_episodes.csv")

    home_scn = C.scenario_of(args.dataset)
    if home_scn is None:
        raise SystemExit("시나리오를 알 수 없는 데이터셋: {}".format(args.dataset))

    dae_path = args.dae_ckpt or C.dae_ckpt_path(args.dataset, args.seed)
    if not os.path.isfile(dae_path):
        raise SystemExit("측정용 DAE가 없습니다: {}\n먼저 v1_dae_ood.py를 실행하세요.".format(dae_path))
    dae, _, dae_meta = C.load_dae_ckpt(dae_path, device)
    thr95, thr99 = dae_meta["thr_p95"], dae_meta["thr_p99"]

    runs = find_runs(args)
    if not runs:
        raise SystemExit("평가할 정책이 없습니다. train_policy.py를 먼저 실행하세요.")
    actors = {name: LoadedActor(path, device) for _, name, path in runs}
    print("[eval] 정책 {}개: {}".format(len(runs), ", ".join(n for _, n, _ in runs)))

    # 조건을 시나리오별로 묶어서 환경을 한 번씩만 만든다
    by_scn = {}
    for cond in args.conditions:
        scn = CONDITION_SCENARIO.get(cond, home_scn)
        by_scn.setdefault(scn, []).append(cond)

    # 설정 파일의 SUMO 시드(random.randrange)를 고정 → 모든 정책이 같은 교통 흐름을 본다
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)

    for scn, conds in by_scn.items():
        print("[eval] 시나리오 {} 준비 (조건: {})".format(scn, ", ".join(conds)))
        env, horizon = make_env(scn)
        warmup_ts = C.WARMUP_TS.get(scn, 0)
        for cond in conds:
            for ep in range(args.episodes):
                ep_seed = args.eval_seed + ep
                for preset, name, _ in runs:
                    set_sumo_seed(env, ep_seed)
                    res = run_episode(env, horizon, warmup_ts, actors[name], cond, args.noise_std,
                                      dae, thr95, thr99, device, ep_seed, args.max_steps)
                    row = {"run": name, "preset": preset, "dataset": args.dataset, "train_seed": args.seed,
                           "condition": cond, "scenario": scn,
                           "noise_std": args.noise_std if cond.startswith(("noise", "denoise")) else 0.0,
                           "episode": ep, "eval_seed": ep_seed}
                    row.update(res)
                    C.append_csv(out_path, [row], FIELDS)
                    print("  {:<10} {:<13} ep{} | steps {:>4} | corr {:>8.1f} | 조기종료 {} | 방문오차 {:.5f}"
                          .format(preset, cond, ep, res["steps"], res["correction_reward"],
                                  res["early_termination"], res["visited_recon_mean"]))
                    sys.stdout.flush()
        try:
            env.terminate()
        except Exception:
            pass

    print("[eval] 완료 → {}".format(out_path))


if __name__ == "__main__":
    main()

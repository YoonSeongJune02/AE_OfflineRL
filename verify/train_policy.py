"""
정책 학습 (SUMO 없이). 학습이 끝나면 정책을 저장하고, 평가는 evaluate.py가 따로 한다.

원래 main_*.py는 15,000 step마다 SUMO로 평가해서 학습 시간 대부분을 평가가 차지했다.
검증에는 최종 정책만 있으면 되므로 평가를 빼고 학습만 한다. 환경을 만들지 않으니
SUMO 포트 충돌도 없고 여러 개를 동시에 돌려도 된다.

프리셋
  cql       : AD4RL 원본 DDPG+CQL (비교 기준)
  dae_orig  : 지금까지 실험한 DAE+CQL 그대로 (λ=1.0 고정, 학습 중 state에 노이즈 0.2를 넣고
              오차 계산, 현재 state 기준, warmup 10000)
  dae_fix   : 의도에 맞게 고친 DAE+CQL
              - 노이즈 없이 오차 계산 (패널티가 state에 따라 일관되게 나오도록)
              - next_state 기준 (행동의 결과로 도착한 상태가 낯설면 그 행동에 패널티)
              - λ = k × (버퍼 reward 표준편차)  (데이터셋마다 reward 스케일에 맞춤)

실행 예:
    python verify/v1_dae_ood.py --dataset highway-NGSIM --seed 5      # DAE 먼저
    CUDA_VISIBLE_DEVICES=0 python verify/train_policy.py --preset cql      --dataset highway-NGSIM --seed 5
    CUDA_VISIBLE_DEVICES=1 python verify/train_policy.py --preset dae_orig --dataset highway-NGSIM --seed 5
    CUDA_VISIBLE_DEVICES=2 python verify/train_policy.py --preset dae_fix  --dataset highway-NGSIM --seed 5
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import common as C
from Algos.DDPG_CQL import DDPGCQL
from Algos.DDPG_CQL_DAE_v2 import DDPGCQL_DAE, soft_update
from Algos.reward_shaping_v2 import RewardShaper
from Utils.utils import ReplayBuffer, set_seed

import torch.nn.functional as F

PRESETS = {
    "cql": {"algo": "cql"},
    "dae_orig": {"algo": "dae", "lambda_mode": "fixed", "lam": 1.0,
                 "shape_noise_std": 0.2, "shape_on": "state", "warmup_steps": 10000},
    "dae_fix": {"algo": "dae", "lambda_mode": "reward_std", "lam_k": 1.0,
                "shape_noise_std": 0.0, "shape_on": "next_state", "warmup_steps": 10000},
}


def parse_args():
    p = argparse.ArgumentParser(description="정책 학습 후 저장 (평가 없음)")
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--tag", default="", help="실행 이름 뒤에 붙일 꼬리표 (예: k0.5)")

    # 원래 main과 같은 기본값
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--itr", type=int, default=15000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--target-update-interval", type=int, default=2)

    # 프리셋 값을 덮어쓰고 싶을 때만 지정
    p.add_argument("--dae-ckpt", default=None, help="기본값: verify_out/checkpoints/dae/<dataset>_seed<seed>.pt")
    p.add_argument("--lam", type=float, default=None, help="lambda_mode=fixed일 때 λ")
    p.add_argument("--lam-k", type=float, default=None, help="lambda_mode=reward_std일 때 λ = k × reward 표준편차")
    p.add_argument("--shape-noise-std", type=float, default=None)
    p.add_argument("--shape-on", choices=["state", "next_state"], default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--threshold-z", type=float, default=1.0)

    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--save-every-epochs", type=int, default=10, help="중간 저장 간격 (0이면 마지막만)")
    p.add_argument("--wandb", action="store_true", help="학습 지표를 WandB에도 기록")
    p.add_argument("--project", default="ae_offlinerl")
    p.add_argument("--group", default="verify")
    return p.parse_args()


class DAEShapedCQL(DDPGCQL_DAE):
    """DDPGCQL_DAE에서 패널티를 state 대신 next_state 기준으로도 계산할 수 있게 한 것.

    나머지(Actor/Critic, CQL 손실, 업데이트 순서)는 원본과 한 줄도 다르지 않다.
    shape_noise_std는 부모의 eval_noise_std 자리에 들어간다(부모는 학습 중 패널티 계산에 이 값을 썼다).
    """

    def __init__(self, *a, **kw):
        self.shape_on = kw.pop("shape_on", "state")
        super(DAEShapedCQL, self).__init__(*a, **kw)

    def train(self, replay_buffer):
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        state, action, next_state, reward, not_done = batch

        target = next_state if self.shape_on == "next_state" else state
        shaped_reward, shape_info = self._compute_shaped_reward(target, reward)

        cql_loss = self.conservative_q_loss(state, action)
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            qft_next = self.critic_target(next_state, next_action)
            next_q = shaped_reward + (1 - not_done) * self.discount * qft_next

        qf = self.critic(state, action)
        critic_loss = cql_loss + 0.5 * F.mse_loss(qf, next_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        actor_loss_val = self.actor_loss(state).mean()
        self.actor_optimizer.zero_grad()
        actor_loss_val.backward()
        self.actor_optimizer.step()

        if self.it % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)
        self.it += 1

        info = {"critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss_val.item()),
                "cql_loss": float(cql_loss.item())}
        info.update(shape_info)
        return info


def cql_train_step(policy, replay_buffer):
    """원본 DDPGCQL.train은 아무것도 반환하지 않아서, 손실 값을 보려고 같은 내용을 다시 쓴 것."""
    batch = replay_buffer.sample(policy.batch)
    batch = [b.to(policy.device) for b in batch]
    state, action, next_state, reward, not_done = batch

    cql_loss = policy.conservative_q_loss(state, action)
    with torch.no_grad():
        next_state_action = policy.actor_target.forward(next_state)
        qft_next = policy.critic_target.forward(next_state, next_state_action)
        next_q = reward + (1 - not_done) * policy.discount * qft_next
    qf = policy.critic.forward(state, action)
    critic_loss = cql_loss + 0.5 * F.mse_loss(qf, next_q)
    policy.critic_optimizer.zero_grad()
    critic_loss.backward(retain_graph=True)
    policy.critic_optimizer.step()

    actor_loss = policy.actor_loss(state).mean()
    policy.actor_optimizer.zero_grad()
    actor_loss.backward()
    policy.actor_optimizer.step()

    if policy.it % policy.target_update_interval == 0:
        soft_update(policy.critic_target, policy.critic, policy.tau)
        soft_update(policy.actor_target, policy.actor, policy.tau)
    policy.it += 1
    return {"critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "cql_loss": float(cql_loss.item())}


def resolve_config(args):
    cfg = dict(PRESETS[args.preset])
    for key, val in [("lam", args.lam), ("lam_k", args.lam_k), ("shape_noise_std", args.shape_noise_std),
                     ("shape_on", args.shape_on), ("warmup_steps", args.warmup_steps)]:
        if val is not None:
            cfg[key] = val
    if args.lam is not None:
        cfg["lambda_mode"] = "fixed"
    elif args.lam_k is not None:
        cfg["lambda_mode"] = "reward_std"
    cfg["threshold_z"] = args.threshold_z
    return cfg


def run_name(args):
    name = "{}_{}_seed{}".format(args.preset, args.dataset, args.seed)
    return name + ("_" + args.tag if args.tag else "")


def policy_dir(name):
    return os.path.join(C.CKPT_DIR, "policy", name)


def save_policy(policy, out_dir, suffix, meta):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    torch.save(policy.actor.state_dict(), os.path.join(out_dir, "actor_{}.pt".format(suffix)))
    torch.save(policy.critic.state_dict(), os.path.join(out_dir, "critic_{}.pt".format(suffix)))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    C.ensure_dirs()
    set_seed(args.seed)
    device = C.get_device()
    args.device = device  # DDPGCQL / DDPGCQL_DAE가 args.device, args.batch 등을 읽는다

    cfg = resolve_config(args)
    name = run_name(args)
    out_dir = policy_dir(name)
    print("[train] {} | device={} | 설정={}".format(name, device, json.dumps(cfg, ensure_ascii=False)))

    # 버퍼 (원래 main과 같은 방식)
    buffer_dir = os.path.join(C.BUFFER_DIR, args.dataset)
    buffer_size = len(np.load(os.path.join(buffer_dir, "reward.npy")))
    replay_buffer = ReplayBuffer(C.STATE_DIM, C.ACTION_DIM, device, buffer_size)
    replay_buffer.load(buffer_dir)
    reward_std = float(np.std(replay_buffer.reward[:replay_buffer.size]))

    meta = {"run": name, "preset": args.preset, "dataset": args.dataset, "seed": args.seed,
            "config": cfg, "reward_std": reward_std, "epochs": args.epochs, "itr": args.itr,
            "batch": args.batch, "scenario": C.scenario_of(args.dataset)}

    if cfg["algo"] == "cql":
        policy = DDPGCQL(args, C.STATE_DIM, C.ACTION_DIM, C.MAX_ACTION, None, args.discount, args.tau)
        step_fn = lambda: cql_train_step(policy, replay_buffer)  # noqa: E731
    else:
        dae_path = args.dae_ckpt or C.dae_ckpt_path(args.dataset, args.seed)
        if not os.path.isfile(dae_path):
            raise SystemExit("DAE 체크포인트가 없습니다: {}\n먼저 실행: python verify/v1_dae_ood.py "
                             "--dataset {} --seed {}".format(dae_path, args.dataset, args.seed))
        dae, _, _ = C.load_dae_ckpt(dae_path, device)
        if cfg["lambda_mode"] == "reward_std":
            lam = cfg["lam_k"] * reward_std
        else:
            lam = cfg["lam"]
        cfg["lambda_used"] = lam
        meta["dae_ckpt"] = dae_path
        shaper = RewardShaper(lambda_max=lam, warmup_steps=cfg["warmup_steps"],
                              threshold_z=cfg["threshold_z"], use_normalization=True, use_tanh=True)
        policy = DAEShapedCQL(args, C.STATE_DIM, C.ACTION_DIM, C.MAX_ACTION, None,
                              dae=dae, reward_shaper=shaper, discount=args.discount, tau=args.tau,
                              eval_noise_std=cfg["shape_noise_std"], shape_on=cfg["shape_on"])
        step_fn = lambda: policy.train(replay_buffer)  # noqa: E731
        print("[train] λ = {:.4f} (reward 표준편차 {:.4f})".format(lam, reward_std))

    if args.wandb:
        import wandb
        wandb.init(project=args.project, group=args.group, name=name, config=meta)

    log_path = os.path.join(C.RESULT_DIR, "train_logs", name + ".csv")
    if os.path.isfile(log_path):
        os.remove(log_path)
    log_fields = ["it", "critic_loss", "actor_loss", "cql_loss", "lambda", "penalty_mean",
                  "penalty_active_frac", "recon_mean", "elapsed_min"]

    total = args.epochs * args.itr
    t0 = time.time()
    acc = {}
    n_acc = 0
    for it in range(total):
        info = step_fn()
        for k, v in info.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n_acc += 1

        if (it + 1) % args.log_every == 0:
            row = {k: v / n_acc for k, v in acc.items()}
            row["it"] = it + 1
            row["elapsed_min"] = (time.time() - t0) / 60.0
            C.append_csv(log_path, [row], log_fields)
            if args.wandb:
                wandb.log({"train/" + k: v for k, v in row.items() if k != "it"}, step=it + 1)
            acc, n_acc = {}, 0

        if (it + 1) % (args.itr * 5) == 0:
            done_frac = (it + 1) / float(total)
            elapsed = (time.time() - t0) / 60.0
            print("[train] {:5.1f}%  경과 {:.1f}분  남은 예상 {:.1f}분".format(
                100 * done_frac, elapsed, elapsed / done_frac * (1 - done_frac)))

        epoch = (it + 1) // args.itr
        if args.save_every_epochs and (it + 1) % args.itr == 0 and epoch % args.save_every_epochs == 0 \
                and epoch != args.epochs:
            save_policy(policy, out_dir, "ep{}".format(epoch), meta)

    meta["train_minutes"] = (time.time() - t0) / 60.0
    save_policy(policy, out_dir, "final", meta)
    print("[train] 저장 완료: {} ({:.1f}분)".format(out_dir, meta["train_minutes"]))
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

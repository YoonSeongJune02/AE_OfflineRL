"""
Main training script for DDPG+CQL+DAE on AD4RL.

Pipeline:
    1. Load offline replay buffer (clean human-driving data).
    2. Pretrain DAE on the buffer with denoising objective.
    3. Train DDPG+CQL with DAE-based reward shaping.
    4. Periodically evaluate in noise-perturbed environment and log
       collision count, cumulative reward, etc., to WandB.
"""

import argparse
import json
import uuid
from copy import deepcopy

import gym
import numpy as np
import torch
import wandb
from tqdm import tqdm

from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env

from Algos.DAE_v2 import DAE, pretrain_dae
from Algos.DDPG_CQL_DAE_v2 import DDPGCQL_DAE
from Algos.reward_shaping_v2 import RewardShaper
from Utils.utils import *  # noqa: F401, F403


# ==========================================================
# Argparse
# ==========================================================
parser = argparse.ArgumentParser(
    description="DDPG+CQL+DAE for robust offline RL on AD4RL.",
)

# Required & flow-related
parser.add_argument("exp_config", type=str)
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--num_cpus", type=int, default=1)
parser.add_argument("--rollout_size", type=int, default=100)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--no_render", action="store_true")

# Dataset / seed / logging
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--load_model", type=str, default=None)
parser.add_argument("--logdir", type=str, default="./results/")

# Fine-tuning / horizon
parser.add_argument("--fine-tune", action="store_true")
parser.add_argument("--num", type=int)
parser.add_argument("--buffers", type=int, default=int(1e6))
parser.add_argument("--horizon", type=int, default=3000)
parser.add_argument("--max-ts", type=int, default=int(1e6))

# Offline RL
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--itr", type=int, default=15000)
parser.add_argument("--num-evaluations", type=int, default=5)

# DDPG
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--discount", type=float, default=0.99)

# CQL
parser.add_argument("--l2_rate", type=float, default=1e-3)
parser.add_argument("--actor_lr", type=float, default=1e-04)
parser.add_argument("--critic_lr", type=float, default=1e-04)
parser.add_argument("--target-update-interval", type=int, default=2)
parser.add_argument("--policy-type")

# DAE: architecture
parser.add_argument("--dae-latent-dim", type=int, default=32)
parser.add_argument("--dae-hidden-dim", type=int, default=128)
parser.add_argument("--dae-dropout", type=float, default=0.1)

# DAE: noise
parser.add_argument("--dae-noise-type", type=str, default="mixed",
                    choices=["gaussian", "masking", "mixed"])
parser.add_argument("--dae-noise-std", type=float, default=0.1)
parser.add_argument("--dae-mask-prob", type=float, default=0.05)

# DAE: pretraining
parser.add_argument("--dae-epochs", type=int, default=100)
parser.add_argument("--dae-lr", type=float, default=1e-3)
parser.add_argument("--dae-weight-decay", type=float, default=1e-5)
parser.add_argument("--dae-val-split", type=float, default=0.1)
parser.add_argument("--dae-patience", type=int, default=10)
parser.add_argument("--dae-batch-size", type=int, default=256)

# Reward shaping
parser.add_argument("--shape-lambda-max", type=float, default=0.5,
                    help="Final penalty coefficient.")
parser.add_argument("--shape-warmup-steps", type=int, default=10000,
                    help="Steps over which lambda ramps from 0. 0 disables warmup.")
parser.add_argument("--shape-threshold-z", type=float, default=1.0,
                    help="Z-score threshold below which no penalty is applied.")
parser.add_argument("--shape-no-normalization", action="store_true",
                    help="Disable running normalization of recon error.")
parser.add_argument("--shape-no-tanh", action="store_true",
                    help="Disable tanh squashing of penalty.")

# Evaluation noise (sim of accident scenario)
parser.add_argument("--eval-noise-std", type=float, default=0.2)

# WandB
parser.add_argument("--project", default="AE_OfflineRL")
parser.add_argument("--group", default="DAE-CQL")
parser.add_argument("--name", default="DDPGCQL_DAE")

args = parser.parse_args()
args.device = torch.device("cpu")
print(args.device)
args.render = not args.no_render


# ==========================================================
# Train / Eval
# ==========================================================
def main(args, replay_buffer, dae: DAE, reward_shaper: RewardShaper) -> None:
    module = __import__("exp_configs.rl.singleagent", fromlist=[args.exp_config])
    module_ma = __import__("exp_configs.rl.multiagent", fromlist=[args.exp_config])

    if hasattr(module, args.exp_config):
        submodule = getattr(module, args.exp_config)
    elif hasattr(module_ma, args.exp_config):
        submodule = getattr(module_ma, args.exp_config)
    else:
        raise ValueError("Unable to find experiment config.")

    flow_params = submodule.flow_params

    import ray
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    agent_cls = get_agent_class("PPO")
    config = deepcopy(agent_cls._default_config)
    config["env_config"]["flow_params"] = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4
    )

    ray.init(num_cpus=16, object_store_memory=200 * 1024 * 1024)

    create_env, gym_name = make_create_env(params=flow_params, version=0)
    register_env(gym_name, create_env)
    agent_cls(env=gym_name, config=config)

    if args.exp_config == "MA_4BL":
        warmup_ts = 900
    elif args.exp_config == "MA_5LC":
        warmup_ts = 125
    elif args.exp_config == "UnifiedRing":
        warmup_ts = 90
    else:
        warmup_ts = 0

    env = gym.make(gym_name)
    env_set_seed(env, args.seed)

    num_inputs, num_actions, max_action = 19, 2, 1.0
    print(env.action_space)
    print(f"state size: {num_inputs}, action size: {num_actions}")

    replay_buffer.load(f"./buffers/{args.dataset}")

    policy = DDPGCQL_DAE(
        args, num_inputs, num_actions, max_action, env.action_space,
        dae=dae,
        reward_shaper=reward_shaper,
        discount=args.discount,
        tau=args.tau,
        eval_noise_std=args.eval_noise_std,
    )

    for it in tqdm(range(args.epochs * args.itr)):
        train_info = policy.train(replay_buffer)

        # Per-step train logging (sparse, every 200 steps)
        if it % 200 == 0:
            wandb.log({f"train/{k}": v for k, v in train_info.items()}, step=it)

        if (it + 1) % args.itr == 0:
            evaluations, velocity, timesteps, collision_counts = [], [], [], []

            for _ in range(args.num_evaluations):
                env.seed(args.seed + 100)
                tot_reward = 0.0
                state, done = env.reset(), False
                episode_vel = []
                ts = 0
                collisions = 0

                while ts <= args.max_ts:
                    if args.render:
                        env.render()

                    state_values = list(state.values())

                    if args.eval_noise_std > 0:
                        state_input = (
                            np.array(state_values)
                            + np.random.randn(len(state_values)) * args.eval_noise_std
                        )
                    else:
                        state_input = state_values

                    action = policy.select_action(state_input)
                    action = {list(state.keys())[0]: action}
                    episode_vel.append(state_values[0])
                    next_state, reward, done, info = env.step(action)
                    tot_reward += list(reward.values())[0]

                    if isinstance(info, dict):
                        collisions += sum(
                            1 for v in info.values()
                            if isinstance(v, dict) and v.get("collision", False)
                        )

                    if done["__all__"]:
                        timesteps.append(
                            env.unwrapped.k.vehicle.get_timestep(
                                env.unwrapped.k.vehicle.get_ids()[1]
                            ) / 100
                        )
                        break

                    state = next_state
                    ts += 1

                velocity.append(np.mean(episode_vel))
                evaluations.append(tot_reward)
                collision_counts.append(collisions)

            eval_reward = float(np.mean(evaluations))
            eval_timestep = float(np.mean(timesteps) - warmup_ts) if timesteps else 0.0
            correction_reward = (
                float(np.mean(np.array(evaluations) + np.array(timesteps) - warmup_ts))
                if timesteps else eval_reward
            )
            avg_collisions = float(np.mean(collision_counts))

            print("-" * 80)
            print(f"# itr: {it} | reward: {eval_reward:.2f} | corr.reward: {correction_reward:.2f}")
            print(f"# collisions: {avg_collisions:.2f} | recon: {train_info.get('recon_mean', 0):.6f} "
                  f"| lambda: {train_info.get('lambda', 0):.4f}")
            print(f"# velocity: {velocity}")
            print(f"# avg episode len: {eval_timestep:.2f}")
            print("-" * 80)

            wandb.log({
                "eval/vanilla_reward": eval_reward,
                "eval/correction_reward": correction_reward,
                "eval/timesteps": eval_timestep,
                "eval/collision_count": avg_collisions,
            }, step=it)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


# ==========================================================
# Entry
# ==========================================================
if __name__ == "__main__":
    seed_list = [5, 6, 7]
    env_list = ["highway-humanlike", "highway-ngsim"]
    args.dataset = env_list[0]
    print(f"--------------------Dataset: {args.dataset}--------------------")

    for j in seed_list:
        args.seed = j
        state_dim, action_dim = 19, 2
        set_seed(args.seed)

        args.name = f"{args.name}-Seed{args.seed}-{args.dataset}-{str(uuid.uuid4())[:8]}"
        wandb_init(vars(args))

        buffer_size = len(np.load(f"./buffers/{args.dataset}/reward.npy"))
        replay_buffer = ReplayBuffer(state_dim, action_dim, args.device, buffer_size)
        replay_buffer.load(f"./buffers/{args.dataset}")

        # ---- 1. DAE pretraining ----
        print(f"\n[Seed {j}] DAE 사전학습 시작...")
        dae = DAE(
            state_dim=state_dim,
            latent_dim=args.dae_latent_dim,
            hidden_dim=args.dae_hidden_dim,
            dropout=args.dae_dropout,
        ).to(args.device)

        dae, dae_history = pretrain_dae(
            dae, replay_buffer, args.device,
            noise_type=args.dae_noise_type,
            noise_std=args.dae_noise_std,
            mask_prob=args.dae_mask_prob,
            epochs=args.dae_epochs,
            batch_size=args.dae_batch_size,
            lr=args.dae_lr,
            weight_decay=args.dae_weight_decay,
            val_split=args.dae_val_split,
            patience=args.dae_patience,
            log_wandb=True,
        )

        wandb.log({
            "dae/best_epoch": dae_history["best_epoch"],
            "dae/final_train_loss": dae_history["train_loss"][-1],
            "dae/final_val_loss": dae_history["val_loss"][-1],
        })

        # ---- 2. Reward shaper ----
        reward_shaper = RewardShaper(
            lambda_max=args.shape_lambda_max,
            warmup_steps=args.shape_warmup_steps,
            threshold_z=args.shape_threshold_z,
            use_normalization=not args.shape_no_normalization,
            use_tanh=not args.shape_no_tanh,
        )

        # ---- 3. CQL training with shaping ----
        print("-" * 53)
        main(args, replay_buffer, dae, reward_shaper)
        wandb.finish()

        args.name = "DDPGCQL_DAE"
        print("-------------------DONE OFFLINE RL-------------------")

        import ray
        ray.shutdown()
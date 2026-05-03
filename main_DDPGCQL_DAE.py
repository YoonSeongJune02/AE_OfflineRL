import os
import sys
import gym
import json
import pickle
import random
import argparse
import numpy as np
from copy import deepcopy

import torch

from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env

from Algos.DDPG_CQL_DAE import DDPGCQL_DAE
from Algos.DAE import DAE, pretrain_dae
from Utils.utils import *
from tqdm import tqdm

import uuid
import wandb

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Parse argument used when running a Flow simulation.",
    epilog="python simulate.py EXP_CONFIG")

# required input parameters
parser.add_argument('exp_config', type=str)
parser.add_argument('--algorithm', type=str, default="PPO")
parser.add_argument('--num_cpus', type=int, default=1)
parser.add_argument('--rollout_size', type=int, default=100)
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--no_render', action='store_true')

# network and dataset setting
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--logdir', type=str, default='./results/')

# Fine tune parameter
parser.add_argument('--fine-tune', action='store_true')
parser.add_argument('--num', type=int)
parser.add_argument('--buffers', type=int, default=int(1e6))
parser.add_argument('--horizon', type=int, default=3000)
parser.add_argument('--max-ts', type=int, default=int(1e6))

# Offline RL parameter
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--itr', type=int, default=15000)
parser.add_argument('--num-evaluations', type=int, default=5)

# DDPG parameter
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--discount', type=float, default=0.99)

# CQL algorithm parameter
parser.add_argument('--l2_rate', type=float, default=1e-3)
parser.add_argument('--actor_lr', type=float, default=1e-04)
parser.add_argument('--critic_lr', type=float, default=1e-04)
parser.add_argument('--target-update-interval', type=int, default=2)
parser.add_argument('--policy-type')

# DAE parameter
parser.add_argument('--dae-latent-dim', type=int, default=32)
parser.add_argument('--dae-noise-std', type=float, default=0.1,
                    help='DAE 사전학습 시 노이즈 강도')
parser.add_argument('--dae-epochs', type=int, default=50,
                    help='DAE 사전학습 epoch 수')
parser.add_argument('--dae-lambda', type=float, default=0.5,
                    help='reward shaping 패널티 강도')
parser.add_argument('--eval-noise-std', type=float, default=0.2,
                    help='평가 시 state에 주입할 노이즈 강도 (0이면 비활성화)')

parser.add_argument('--project', default='AE_OfflineRL')
parser.add_argument('--group', default='DAE-CQL')
parser.add_argument('--name', default='DDPGCQL_DAE')

args = parser.parse_args()
args.device = torch.device("cpu")
print(args.device)
args.render = not args.no_render


def main(args, replay_buffer, dae):
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[args.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[args.exp_config])

    if hasattr(module, args.exp_config):
        submodule = getattr(module, args.exp_config)
        multiagent = False
    elif hasattr(module_ma, args.exp_config):
        submodule = getattr(module_ma, args.exp_config)
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    flow_params = submodule.flow_params

    import ray
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    alg_run = "PPO"
    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    ray.init(num_cpus=16, object_store_memory=200 * 1024 * 1024)

    create_env, gym_name = make_create_env(params=flow_params, version=0)
    register_env(gym_name, create_env)
    agent = agent_cls(env=gym_name, config=config)

    # warmup correction
    if args.exp_config == 'MA_4BL':
        warmup_ts = 900
    elif args.exp_config == 'MA_5LC':
        warmup_ts = 125
    elif args.exp_config == 'UnifiedRing':
        warmup_ts = 90

    env = gym.make(gym_name)
    env_set_seed(env, args.seed)

    num_inputs = 19
    num_actions = 2
    max_action = 1.0
    print(env.action_space)
    print('state size:', num_inputs)
    print('action size:', num_actions)

    buffer_name = f"{args.dataset}"
    replay_buffer.load(f"./buffers/{buffer_name}")

    # DAE + CQL 결합 모델 초기화
    policy = DDPGCQL_DAE(
        args, num_inputs, num_actions, max_action, env.action_space,
        dae=dae,
        discount=args.discount,
        tau=args.tau,
        dae_lambda=args.dae_lambda,
        noise_std=args.eval_noise_std,
    )

    reward_list = []
    for it in tqdm(range(args.epochs * args.itr)):
        recon_error = policy.train(replay_buffer)

        if (it + 1) % args.itr == 0:
            evaluations = []
            velocity = []
            timesteps = []
            collision_counts = []

            for _ in range(args.num_evaluations):
                env.seed(args.seed + 100)
                tot_reward = 0.
                state, done = env.reset(), False

                episode_vel = []
                ts = 0
                collisions = 0

                while ts <= args.max_ts:
                    if args.render:
                        env.render()

                    state_values = list(state.values())

                    # 평가 시 노이즈 주입 (eval_noise_std > 0이면 활성화)
                    if args.eval_noise_std > 0:
                        state_noisy = np.array(state_values) + np.random.randn(len(state_values)) * args.eval_noise_std
                    else:
                        state_noisy = state_values

                    action = policy.select_action(state_noisy)
                    action = {list(state.keys())[0]: action}
                    episode_vel.append(state_values[0])
                    next_state, reward, done, info = env.step(action)
                    tot_reward += list(reward.values())[0]

                    # 충돌 감지 (info에 collision 정보가 있는 경우)
                    if isinstance(info, dict):
                        collisions += sum(1 for v in info.values()
                                          if isinstance(v, dict) and v.get('collision', False))

                    if done['__all__']:
                        timesteps.append(
                            env.unwrapped.k.vehicle.get_timestep(
                                env.unwrapped.k.vehicle.get_ids()[1]) / 100)
                        break

                    state = next_state
                    ts += 1

                velocity.append(np.mean(episode_vel))
                evaluations.append(tot_reward)
                collision_counts.append(collisions)

            eval_reward = np.mean(evaluations)
            eval_timestep = np.mean(timesteps) - warmup_ts
            correction_reward = np.mean(np.array(evaluations) + np.array(timesteps) - warmup_ts)
            avg_collisions = np.mean(collision_counts)

            print('----------------------------------------------------------------------------------------')
            print(f'# itr: {it} | avg.reward: {eval_reward:.2f} | cor.reward: {correction_reward:.2f}')
            print(f'# avg.collisions: {avg_collisions:.2f} | recon_error: {recon_error:.6f}')
            print(f'# velocity: {velocity} over {args.num_evaluations} evaluations')
            print(f'# avg episode len: {eval_timestep:.2f}')
            print('----------------------------------------------------------------------------------------')

            wandb.log({
                "vanilla_reward": eval_reward,
                "correction_reward": correction_reward,
                "timesteps": eval_timestep,
                "collision_count": avg_collisions,     # baseline과 비교할 핵심 지표
                "recon_error": recon_error,            # DAE 재구성오차 모니터링
                "dae_lambda": args.dae_lambda,
            }, step=it)

            reward_list.append(eval_reward)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config['project'],
        group=config['group'],
        name=config['name'],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


if __name__ == "__main__":
    seed_list = [5, 6, 7]
    env_list = ['highway-humanlike', 'highway-ngsim']
    args.dataset = env_list[0]
    print(f'--------------------Dataset: {args.dataset}--------------------')

    for j in seed_list:
        args.seed = j

        state_dim = 19
        action_dim = 2
        buffer_name = f"{args.dataset}"
        set_seed(args.seed)

        args.name = f"{args.name}-Seed{args.seed}-{args.dataset}-{str(uuid.uuid4())[:8]}"
        config = vars(args)
        wandb_init(config)

        buffer_size = len(np.load(f"./buffers/{buffer_name}/reward.npy"))
        replay_buffer = ReplayBuffer(state_dim, action_dim, args.device, buffer_size)
        replay_buffer.load(f"./buffers/{buffer_name}")

        # DAE 사전학습 (정상 데이터로)
        print(f"[Seed {j}] DAE 사전학습 시작...")
        dae = DAE(state_dim=state_dim, latent_dim=args.dae_latent_dim).to(args.device)
        dae = pretrain_dae(
            dae, replay_buffer, args.device,
            noise_std=args.dae_noise_std,
            epochs=args.dae_epochs,
            batch_size=args.batch,
        )

        # WandB에 DAE 사전학습 완료 로그
        wandb.log({"dae_pretrain": "done", "dae_noise_std": args.dae_noise_std})

        print('-----------------------------------------------------')
        main(args, replay_buffer, dae)
        wandb.finish()

        args.name = 'DDPGCQL_DAE'
        print('-------------------DONE OFFLINE RL-------------------')

        import ray
        ray.shutdown()
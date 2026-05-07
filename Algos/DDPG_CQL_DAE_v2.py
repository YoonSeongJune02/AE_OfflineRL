"""
DDPG + CQL with DAE-based reward shaping for robust offline RL.

The base DDPG+CQL algorithm follows the AD4RL implementation. The only
modification is reward shaping: at each training step, the reward used in
the Q-target is augmented by a penalty proportional to the DAE reconstruction
error of the (optionally noise-perturbed) state.

The shaping logic is delegated to RewardShaper (see reward_shaping.py).
"""

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.exploration import OUNoise
from Algos.model import *  # noqa: F401, F403  (kept for compatibility with original)
from Algos.DAE_v2 import DAE
from Algos.reward_shaping_v2 import RewardShaper


# ==========================================================
# Actor / Critic (identical to baseline DDPG_CQL)
# ==========================================================
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


# ==========================================================
# Algorithm
# ==========================================================
class DDPGCQL_DAE:
    """DDPG + CQL with DAE reward shaping.

    Args:
        args: argparse Namespace with attributes (device, batch, target_update_interval,
              actor_lr, critic_lr).
        state_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        max_action: Action bound (symmetric).
        action_space: Gym action space (kept for compatibility, unused here).
        dae: A pretrained DAE module.
        reward_shaper: Pre-configured RewardShaper instance.
        discount: Discount factor.
        tau: Soft update coefficient for target networks.
        eval_noise_std: Std of noise to inject into states during reward shaping.
                        Set to 0.0 to disable (use raw replay-buffer states).
    """

    def __init__(
        self,
        args,
        state_dim: int,
        action_dim: int,
        max_action: float,
        action_space,
        dae: DAE,
        reward_shaper: RewardShaper,
        discount: float = 0.99,
        tau: float = 0.005,
        eval_noise_std: float = 0.0,
    ) -> None:
        self.device = args.device
        self.batch = args.batch
        self.target_update_interval = args.target_update_interval

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau

        self.dae = dae
        self.reward_shaper = reward_shaper
        self.eval_noise_std = eval_noise_std

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.exp = OUNoise(action_dim)
        self.it = 0

    def select_action(self, obs) -> torch.Tensor:
        obs = torch.FloatTensor(obs).to(self.device)
        return self.actor(obs).cpu().data.numpy().flatten()

    def select_exp_action(self, obs) -> torch.Tensor:
        obs = torch.FloatTensor(obs).to(self.device)
        noise = self.exp.noise()
        return self.actor(obs).cpu().data.numpy().flatten() + 0.05 * noise

    def conservative_q_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pol_a = self.actor(obs)
        pol_q = self.critic(obs, pol_a)
        beh_q = self.critic(obs, action)
        return pol_q.mean() - beh_q.mean()

    def actor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.actor(obs)
        return -self.critic(obs, action)

    def _compute_shaped_reward(
        self,
        state: torch.Tensor,
        reward: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Inject optional noise, compute DAE recon error, and shape reward."""
        if self.eval_noise_std > 0:
            noisy_state = state + torch.randn_like(state) * self.eval_noise_std
        else:
            noisy_state = state

        recon_error = self.dae.reconstruction_error(noisy_state)
        shaped_reward, info = self.reward_shaper.shape(reward, recon_error)
        return shaped_reward, info

    def train(self, replay_buffer) -> Dict[str, float]:
        """Run one training step. Returns diagnostics dict for logging."""
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        state, action, next_state, reward, not_done = batch

        shaped_reward, shape_info = self._compute_shaped_reward(state, reward)

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

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss_val.item()),
            "cql_loss": float(cql_loss.item()),
            **shape_info,
        }

    def save(self, filename: str, ep) -> None:
        torch.save(self.actor.state_dict(), f"{filename}_{ep}_actor")
        torch.save(self.actor_target.state_dict(), f"{filename}_{ep}_actor_target")
        torch.save(self.critic.state_dict(), f"{filename}_{ep}_critic")
        torch.save(self.critic_target.state_dict(), f"{filename}_{ep}_critic_target")
        torch.save(self.dae.state_dict(), f"{filename}_{ep}_dae")
        torch.save(self.reward_shaper.state_dict(), f"{filename}_{ep}_shaper")

    def load(self, filename: str, ep) -> None:
        self.actor.load_state_dict(torch.load(f"{filename}_{ep}_actor"))
        self.actor_target.load_state_dict(torch.load(f"{filename}_{ep}_actor_target"))
        self.critic.load_state_dict(torch.load(f"{filename}_{ep}_critic"))
        self.critic_target.load_state_dict(torch.load(f"{filename}_{ep}_critic_target"))
        self.dae.load_state_dict(torch.load(f"{filename}_{ep}_dae"))
        self.reward_shaper.load_state_dict(torch.load(f"{filename}_{ep}_shaper"))


def soft_update(target_net: nn.Module, net: nn.Module, tau: float) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
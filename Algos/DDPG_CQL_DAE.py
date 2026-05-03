import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.exploration import OUNoise
from Algos.model import *
from Algos.DAE import DAE
from tqdm import tqdm


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DDPGCQL_DAE(object):
    def __init__(self,
                 args,
                 state_dim,
                 action_dim,
                 max_action,
                 action_space,
                 dae,                       # 사전학습된 DAE
                 discount=0.99,
                 tau=0.005,
                 dae_lambda=0.5,            # reward shaping 강도
                 noise_std=0.0,             # 평가 시 노이즈 주입 강도 (0이면 비활성화)
                 ):

        self.device = args.device
        self.batch = args.batch
        self.target_update_interval = args.target_update_interval

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau

        # DAE reward shaping 파라미터
        self.dae = dae
        self.dae_lambda = dae_lambda
        self.noise_std = noise_std

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.exp = OUNoise(action_dim)
        self.it = 0

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        return self.actor(obs).cpu().data.numpy().flatten()

    def select_exp_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        noise = self.exp.noise()
        return self.actor(obs).cpu().data.numpy().flatten() + 0.05 * noise

    def conservative_q_loss(self, obs, action):
        pol_a = self.actor.forward(obs)
        pol_q = self.critic.forward(obs, pol_a)
        beh_q = self.critic.forward(obs, action)
        return pol_q.mean() - beh_q.mean()

    def actor_loss(self, obs):
        action = self.actor.forward(obs)
        return -self.critic.forward(obs, action)

    def _shape_reward(self, state, reward):
        """
        DAE 재구성오차를 패널티로 추가한 shaped reward 반환.
        노이즈 주입(noise_std > 0)이 설정된 경우 state에 노이즈를 추가하여
        DAE가 이상 상태를 감지하도록 함.
        """
        if self.noise_std > 0:
            noisy_state = state + torch.randn_like(state) * self.noise_std
        else:
            noisy_state = state

        recon_error = self.dae.reconstruction_error(noisy_state)  # (batch, 1)
        shaped_reward = reward - self.dae_lambda * recon_error
        return shaped_reward, recon_error.mean().item()

    def train(self, replay_buffer):
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        (state, action, next_state, reward, not_done) = batch

        # DAE reward shaping 적용
        shaped_reward, recon_error_mean = self._shape_reward(state, reward)

        cql_loss = self.conservative_q_loss(state, action)

        with torch.no_grad():
            next_state_action = self.actor_target.forward(next_state)
            qft_next = self.critic_target.forward(next_state, next_state_action)
            next_q = shaped_reward + (1 - not_done) * self.discount * qft_next  # shaped reward 사용

        qf = self.critic.forward(state, action)
        critic_loss = cql_loss + 0.5 * F.mse_loss(qf, next_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        actor_loss = self.actor_loss(state).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.it % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        self.it += 1
        return recon_error_mean

    def save(self, filename, ep):
        torch.save(self.actor.state_dict(), filename + f'_{ep}' + '_actor')
        torch.save(self.actor_target.state_dict(), filename + f'_{ep}' + '_actor_target')
        torch.save(self.critic.state_dict(), filename + f'_{ep}' + '_critic')
        torch.save(self.critic_target.state_dict(), filename + f'_{ep}' + '_critic_target')
        torch.save(self.dae.state_dict(), filename + f'_{ep}' + '_dae')

    def load(self, filename, ep):
        self.actor.load_state_dict(torch.load(filename + f'_{ep}' + '_actor'))
        self.actor_target.load_state_dict(torch.load(filename + f'_{ep}' + '_actor_target'))
        self.critic.load_state_dict(torch.load(filename + f'_{ep}' + '_critic'))
        self.critic_target.load_state_dict(torch.load(filename + f'_{ep}' + '_critic_target'))
        self.dae.load_state_dict(torch.load(filename + f'_{ep}' + '_dae'))


def soft_update(target_net, net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
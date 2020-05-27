from model import QNet, PiNet
import torch
from torch import nn
import torch.nn.functional as F
from alg import Algorithm
from utils import soft_update, RandomNoise


class TD3(Algorithm):

    def __init__(self, *largs, **kwargs):
        super(TD3, self).__init__(*largs, **kwargs)

        pi_net = PiNet(self.ns, self.na)
        self.pi_net = pi_net.to(self.device)

        pi_target = PiNet(self.ns, self.na)
        self.pi_target = pi_target.to(self.device)
        self.load_state_dict(self.pi_target, self.pi_net.state_dict())

        q_net_1 = QNet(self.ns, self.na)
        self.q_net_1 = q_net_1.to(self.device)

        q_target_1 = QNet(self.ns, self.na)
        self.q_target_1 = q_target_1.to(self.device)
        self.load_state_dict(self.q_target_1, self.q_net_1.state_dict())

        q_net_2 = QNet(self.ns, self.na)
        self.q_net_2 = q_net_2.to(self.device)

        q_target_2 = QNet(self.ns, self.na)
        self.q_target_2 = q_target_2.to(self.device)
        self.load_state_dict(self.q_target_2, self.q_net_2.state_dict())

        self.optimizer_q_1 = torch.optim.Adam(self.q_net_1.parameters(), lr=self.lr_q, betas=(0.9, 0.999))

        self.optimizer_q_2 = torch.optim.Adam(self.q_net_2.parameters(), lr=self.lr_q, betas=(0.9, 0.999))

        self.optimizer_p = torch.optim.Adam(self.pi_net.parameters(), lr=self.lr_p, betas=(0.9, 0.999))

        self.noise = RandomNoise(torch.zeros(1, self.na).to(self.device), self.epsilon)

    def play(self, env, evaluate=False):

        if env.k == 0:
            self.noise.reset()

        if evaluate:
            with torch.no_grad():
                a = self.pi_net(env.s)

        elif self.env_steps >= self.warmup_steps:
            with torch.no_grad():
                a = self.pi_net(env.s) + self.noise()
            a = torch.clamp(a, min=-1, max=1)
        else:
            a = None

        state = env(a)
        return state

    def replay_buffer_training(self, sample, train_results, n):

        s, a, r, t, stag = [sample[k] for k in ['s', 'a', 'r', 't', 'stag']]

        self.train_mode()

        with torch.no_grad():
            pi_tag = self.pi_target(stag)

            noise = (torch.randn_like(pi_tag) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            pi_tag = (pi_tag + noise).clamp(-1, 1)

            q_target_1 = self.q_target_1(stag, pi_tag)
            q_target_2 = self.q_target_2(stag, pi_tag)

        q_target = torch.min(q_target_1, q_target_2)
        g = r + (1 - t) * self.gamma ** self.n_steps * q_target

        qa = self.q_net_1(s, a)
        loss_q = F.mse_loss(qa, g, reduction='mean')

        self.optimizer_q_1.zero_grad()
        loss_q.backward()
        if self.clip_q:
            nn.utils.clip_grad_norm(self.q_net_1.parameters(), self.clip_q)
        self.optimizer_q_1.step()

        qa = self.q_net_2(s, a)
        loss_q = F.mse_loss(qa, g, reduction='mean')

        self.optimizer_q_2.zero_grad()
        loss_q.backward()
        if self.clip_q:
            nn.utils.clip_grad_norm(self.q_net_2.parameters(), self.clip_q)
        self.optimizer_q_2.step()

        if not n % self.td3_delayed_policy_update:

            pi = self.pi_net(s)

            v = self.q_net_1(s, pi)
            loss_p = (-v).mean()

            self.optimizer_p.zero_grad()
            loss_p.backward()
            if self.clip_p:
                nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
            self.optimizer_p.step()

            train_results['scalar']['objective'].append(float(-loss_p))

            soft_update(self.pi_net, self.pi_target, self.tau)
            soft_update(self.q_net_1, self.q_target_1, self.tau)
            soft_update(self.q_net_2, self.q_target_2, self.tau)

        train_results['scalar']['loss_q'].append(float(loss_q))

        return train_results

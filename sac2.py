from model import MultipleOptimizer, QNet, PiNet
from config import args, exp
import torch
from torch import nn
import torch.nn.functional as F
from sampler import UniversalBatchSampler, HexDataset
from alg import Algorithm
import itertools
import numpy as np
import math
from loguru import logger
from collections import defaultdict
from tqdm import tqdm
import math
import scipy.stats as sts
import copy
from apex import amp
from operator import itemgetter
from collections import namedtuple
from utils import soft_update, OrnsteinUhlenbeckActionNoise, RandomNoise


class SAC2(Algorithm):

    def __init__(self, env):
        super(SAC2, self).__init__()

        self.env = env
        n_a = env.action_space.shape[0]
        n_s = env.observation_space.shape[0]

        pi_net = PiNet(n_s, n_a, distribution='Normal')
        self.pi_net = pi_net.to(self.device)

        q_net_1 = QNet(n_s, n_a)
        self.q_net_1 = q_net_1.to(self.device)

        q_target_1 = QNet(n_s, n_a)
        self.q_target_1 = q_target_1.to(self.device)
        self.load_state_dict(self.q_target_1, self.q_net_1.state_dict())

        q_net_2 = QNet(n_s, n_a)
        self.q_net_2 = q_net_2.to(self.device)

        q_target_2 = QNet(n_s, n_a)
        self.q_target_2 = q_target_2.to(self.device)
        self.load_state_dict(self.q_target_2, self.q_net_2.state_dict())

        self.optimizer_q_1 = torch.optim.Adam(self.q_net_1.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                              weight_decay=self.weight_decay_q)

        self.optimizer_q_2 = torch.optim.Adam(self.q_net_2.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                              weight_decay=self.weight_decay_q)

        self.optimizer_p = torch.optim.Adam(self.pi_net.parameters(), lr=self.lr_p, betas=(0.9, 0.999),
                                    weight_decay=self.weight_decay_p)

        self.sample = self.actor_rb
        if self.entropy_tunning:
            self.target_entropy = -torch.prod(torch.Tensor(n_a).to(self.device)).item()
            self.log_alpha = torch.tensor([0.], requires_grad=True, device=self.device)
            self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=args.lr_q)

    def play(self):

        if self.env_steps >= self.warmup_steps:
            with torch.no_grad():
                a = self.pi_net(self.env.s)
        else:
            a = None

        state = self.env(a)
        return state

    def train(self):

        results = defaultdict(lambda: defaultdict(list))

        for i, sample in enumerate(self.sample()):
            i += 1

            s, a, r, t, stag = [sample[k] for k in ['s', 'a', 'r', 't', 'stag']]

            self.train()

            with torch.no_grad():
                pi_tag = self.pi_net(stag)
                log_pi_tag = self.pi_net.log_prob(pi_tag).sum(dim=1)
                q_target_1 = self.q_target_1(stag, pi_tag)
                q_target_2 = self.q_target_2(stag, pi_tag)

            q_target = torch.min(q_target_1, q_target_2) - float(self.alpha) * log_pi_tag
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

            if not i % self.delayed_policy_update:

                pi = self.pi_net(s)
                qa_1 = self.q_net_1(s, pi)
                qa_2 = self.q_net_2(s, pi)
                qa = torch.min(qa_1, qa_2)
                log_pi = self.pi_net.log_prob(pi).sum(dim=1)

                loss_p = (self.alpha * log_pi - qa).mean()

                self.optimizer_p.zero_grad()
                loss_p.backward()

                if self.clip_p:
                    nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
                self.optimizer_p.step()

                # alpha loss
                if self.entropy_tunning:
                    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                    self.optimizer_alpha.zero_grad()
                    alpha_loss.backward()
                    self.optimizer_alpha.step()

                    self.alpha = self.log_alpha.exp()

                results['scalar']['q_est'].append(float(-loss_p))

            results['scalar']['loss_q'].append(float(loss_q))

            soft_update(self.q_net_1, self.q_target_1, self.tau)
            soft_update(self.q_net_2, self.q_target_2, self.tau)

            # if not n % self.target_update:
            #     self.load_state_dict(self.pi_target, self.pi_net.state_dict())
            #     self.load_state_dict(self.q_target, self.q_net.state_dict())

            if not i % self.train_epoch:

                statistics = self.env.get_stats()
                for k, v in statistics.items():
                    for ki, vi in v.items():
                        results[k][ki] = vi

                results['scalar']['rb'] = self.replay_buffer.size
                results['scalar']['env-steps'] = self.env_steps
                results['scalar']['episodes'] = self.episodes
                results['scalar']['train-steps'] = i

                yield results
                results = defaultdict(lambda: defaultdict(list))

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

Sample = namedtuple('Sample', ('s', 'a', 'r', 't', 'stag'))


class DDPG(Algorithm):

    def __init__(self, env):
        super(DDPG, self).__init__()

        self.env = env
        n_a = env.action_space.shape[0]
        n_s = env.observation_space.shape[0]

        pi_net = PiNet(n_s, n_a)
        self.pi_net = pi_net.to(self.device)

        pi_target = PiNet(n_s, n_a)
        self.pi_target = pi_target.to(self.device)
        self.load_state_dict(self.pi_target, self.pi_net.state_dict())

        q_net = QNet(n_s, n_a)
        self.q_net = q_net.to(self.device)

        q_target = QNet(n_s, n_a)
        self.q_target = q_target.to(self.device)
        self.load_state_dict(self.q_target, self.q_net.state_dict())

        # sparse_parameters = [p for n, p in filter(lambda x: 'embedding' in x[0], self.q_net.named_parameters())]
        # opt_sparse = torch.optim.SparseAdam(sparse_parameters, lr=self.lr_q * 100,
        #                                     betas=(0.9, 0.999), eps=1e-04)
        # opt_dense = torch.optim.Adam(list(set(self.q_net.parameters()).difference(sparse_parameters)), lr=self.lr_q, betas=(0.9, 0.999),
        #                              eps=1e-04,  weight_decay=self.weight_decay)
        # self.optimizer_q = MultipleOptimizer(opt_sparse, opt_dense)
        #
        # sparse_parameters = [p for n, p in filter(lambda x: 'embedding' in x[0], self.pi_net.named_parameters())]
        # opt_sparse = torch.optim.SparseAdam(sparse_parameters, lr=self.lr_p * 100,
        #                                     betas=(0.9, 0.999), eps=1e-04)
        # opt_dense = torch.optim.Adam(list(set(self.pi_net.parameters()).difference(sparse_parameters)), lr=self.lr_p, betas=(0.9, 0.999),
        #                              eps=1e-04,  weight_decay=self.weight_decay)
        # self.optimizer_p = MultipleOptimizer(opt_sparse, opt_dense)

        self.optimizer_q = torch.optim.Adam(self.q_net.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                     weight_decay=1e-2)

        # eps = 1e-04,
        self.optimizer_p = torch.optim.Adam(self.pi_net.parameters(), lr=self.lr_p, betas=(0.9, 0.999),
                                    weight_decay=0)

        self.noise = OrnsteinUhlenbeckActionNoise(torch.zeros(1, n_a).to(self.device),
                                                  self.epsilon * torch.ones(1, n_a).to(self.device))
        self.sample = self.actor_rb

    def play(self):

        if self.env.k == 0:
            self.noise.reset()

        noise = self.noise()
        with torch.no_grad():

            if self.env_steps >= self.warmup_steps:
                a = self.pi_net(self.env.s) + noise
            else:
                a = noise

        state = self.env(torch.clamp(a, min=-1, max=1))
        return state

    def train(self):

        results = defaultdict(lambda: defaultdict(list))

        for i, (s, a, r, t, stag) in tqdm(enumerate(self.sample())):
            i += 1
            self.train()
            self.optimizer_q.zero_grad()
            self.optimizer_p.zero_grad()

            with torch.no_grad():
                pi_tag = self.pi_target(stag)
                q_target = self.q_target(stag, pi_tag)

            g = r + (1 - t) * self.gamma ** self.n_steps * q_target

            qa = self.q_net(s, a)
            loss_q = F.mse_loss(qa, g, reduction='mean')

            loss_q.backward()
            if self.clip_q:
                nn.utils.clip_grad_norm(self.q_net.parameters(), self.clip_q)
            self.optimizer_q.step()

            if not i % self.delayed_policy_update:

                pi = self.pi_net(s)

                if self.env_steps >= self.warmup_steps:

                    v = self.q_net(s, pi)
                    loss_p = (-v).mean()
                else:

                    loss_p = F.smooth_l1_loss(pi, a)

                loss_p.backward()
                if self.clip_p:
                    nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
                self.optimizer_p.step()

                results['scalar']['q_est'].append(float(-loss_p))

                soft_update(self.pi_net, self.pi_target, self.tau)

            results['scalar']['loss_q'].append(float(loss_q))

            soft_update(self.q_net, self.q_target, self.tau)

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

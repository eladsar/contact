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


class SACV(Algorithm):

    def __init__(self, env):
        super(SACV, self).__init__()

        self.env = env
        n_a = env.action_space.shape[0]
        n_s = env.observation_space.shape[0]

        pi_net = PiNet(n_s, n_a, distribution='Normal')
        self.pi_net = pi_net.to(self.device)

        q_net_1 = QNet(n_s, n_a)
        self.q_net_1 = q_net_1.to(self.device)

        q_net_2 = QNet(n_s, n_a)
        self.q_net_2 = q_net_2.to(self.device)

        v_net = QNet(n_s, 0)
        self.v_net = v_net.to(self.device)

        v_target = QNet(n_s, 0)
        self.v_target = v_target.to(self.device)
        self.load_state_dict(self.v_target, self.v_net.state_dict())

        self.optimizer_q_1 = torch.optim.Adam(self.q_net_1.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                     weight_decay=1e-2)

        self.optimizer_q_2 = torch.optim.Adam(self.q_net_2.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                     weight_decay=1e-2)

        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                     weight_decay=1e-2)

        # eps = 1e-04,
        self.optimizer_p = torch.optim.Adam(self.pi_net.parameters(), lr=self.lr_p, betas=(0.9, 0.999),
                                    weight_decay=0)

        self.sample = self.actor_rb

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

        for i, sample in tqdm(enumerate(self.sample())):
            i += 1

            s, a, r, t, stag, beta, w = [sample[k] for k in ['s', 'a', 'r', 't', 'stag']]

            self.train_mode()

            with torch.no_grad():
                v_target = self.v_target(stag)

            g = r + (1 - t) * self.gamma ** self.n_steps * v_target

            qa_1 = self.q_net_1(s, a)
            loss_q_1 = F.mse_loss(qa_1, g, reduction='mean')

            qa_2 = self.q_net_2(s, a)
            loss_q_2 = F.mse_loss(qa_2, g, reduction='mean')

            a_sample = self.pi_net(s)
            log_prob = self.pi_net.log_prob(a_sample)

            # v loss
            with torch.no_grad():

                qa_target_1 = self.q_net_1(s, a_sample.detach())
                qa_target_2 = self.q_net_2(s, a_sample.detach())

                qa_target = torch.min(qa_target_1, qa_target_2)

            g = qa - log_prob.detach()
            v = self.v_net(s)
            loss_q = F.mse_loss(v, g, reduction='mean')

            if not i % self.delayed_policy_update:

                pi = self.pi_net(s)
                qa_1 = self.q_target_1(s, pi)
                qa_2 = self.q_target_2(s, pi)
                qa = torch.min(qa_1, qa_2)
                log_pi = self.pi_net.log_prob(pi).sum(dim=1)

                loss_p = (self.alpha * log_pi - qa).mean()

                self.optimizer_p.zero_grad()
                loss_p.backward()

                if self.clip_p:
                    nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
                self.optimizer_p.step()

                results['scalar']['q_est'].append(float(-loss_p))

            self.optimizer_q_1.zero_grad()
            loss_q_1.backward()
            if self.clip_q:
                nn.utils.clip_grad_norm(self.q_net_1.parameters(), self.clip_q)
            self.optimizer_q_1.step()

            self.optimizer_q_2.zero_grad()
            loss_q_2.backward()
            if self.clip_q:
                nn.utils.clip_grad_norm(self.q_net_2.parameters(), self.clip_q)
            self.optimizer_q_2.step()

            results['scalar']['loss_q'].append(float(loss_q))

            soft_update(self.v_net, self.v_target, self.tau)

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

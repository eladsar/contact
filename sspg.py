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
import torch.autograd as autograd


class SSPG(Algorithm):

    def __init__(self, *largs, **kwargs):
        super(SSPG, self).__init__(*largs, **kwargs)

        pi_net = PiNet(self.ns, self.na, distribution='Normal')
        self.pi_net = pi_net.to(self.device)

        pi_target = PiNet(self.ns, self.na, distribution='Normal')
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

        self.optimizer_q_1 = torch.optim.Adam(self.q_net_1.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                     weight_decay=self.weight_decay_q)

        self.optimizer_q_2 = torch.optim.Adam(self.q_net_2.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                     weight_decay=self.weight_decay_q)

        self.optimizer_p = torch.optim.Adam(self.pi_net.parameters(), lr=self.lr_p, betas=(0.9, 0.999),
                                    weight_decay=self.weight_decay_p)

    def play(self, evaluate=False):

        env = self.env_eval if evaluate else self.env_train

        self.pi_net(self.env_eval.s)
        a = self.pi_net.sample(expected_value=evaluate)
        state = env(a)

        return state

    def offline_training(self, sample, train_results, n):

        s, a, r, t, stag = [sample[k] for k in ['s', 'a', 'r', 't', 'stag']]

        self.train()

        with torch.no_grad():
            self.pi_target(stag)
            pi_tag_1 = self.pi_target.sample(self.rbi_samples)
            pi_tag_2 = self.pi_target.sample(self.rbi_samples)
            q_target_1 = self.q_target_1(stag, pi_tag_1).mean(dim=0)
            q_target_2 = self.q_target_2(stag, pi_tag_2).mean(dim=0)

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

        if not n % self.delayed_policy_update:

            self.pi_net(s)
            pi = self.pi_net.rsample(self.rbi_samples)

            qa_1 = self.q_net_1(s, pi).mean(dim=0)
            qa_2 = self.q_net_2(s, pi).mean(dim=0)
            qa = torch.min(qa_1, qa_2)

            loss_p = (- qa).mean()

            loss_p -= 1e-3 * self.pi_net.entropy().sum(dim=-1).mean()

            self.optimizer_p.zero_grad()
            loss_p.backward()
            if self.clip_p:
                nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
            self.optimizer_p.step()

            train_results['scalar']['q_est'].append(float(-loss_p))
            soft_update(self.pi_net, self.pi_target, self.tau)

        train_results['scalar']['loss_q'].append(float(loss_q))

        soft_update(self.q_net_1, self.q_target_1, self.tau)
        soft_update(self.q_net_2, self.q_target_2, self.tau)

        return train_results


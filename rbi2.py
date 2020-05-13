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

a_lr = 0.05


def max_reroute(s, pi_net, q_net, n=100, cmin=0.5, cmax=1.5, greed=0.1, epsilon=0.01):

    pi_net(s)

    with torch.no_grad():
        beta = pi_net.sample(n)

    _, b, na = beta.shape

    s = s.unsqueeze(0).expand(n, *s.shape)
    s = s.view(n * b, *s.shape[2:])
    beta = beta.view(n * b, na)

    beta = autograd.Variable(beta.data, requires_grad=True)
    q = q_net(s, beta)

    gradients = autograd.grad(outputs=q, inputs=beta, grad_outputs=torch.cuda.FloatTensor(q.size()).fill_(1.),
                              create_graph=False, retain_graph=False, only_inputs=True)[0]

    # q.mean().backward()

    beta = beta.data + a_lr * gradients
    w = torch.ones_like(beta).unsqueeze(0).permute(0, 2, 1)

    a = beta[torch.randint(n, size=(1,))]

    beta = beta.unsqueeze(0).permute(0, 2, 1)

    return a, (beta, w)


class RBI2(Algorithm):

    def __init__(self, env):
        super(RBI2, self).__init__()

        self.env = env
        na = env.action_space.shape[0]
        ns = env.observation_space.shape[0]
        self.na = na

        pi_net = PiNet(ns, na, distribution='Normal')
        self.pi_net = pi_net.to(self.device)

        pi_target = PiNet(ns, na, distribution='Normal')
        self.pi_target = pi_target.to(self.device)
        self.load_state_dict(self.pi_target, self.pi_net.state_dict())

        q_net_1 = QNet(ns, na)
        self.q_net_1 = q_net_1.to(self.device)

        q_target_1 = QNet(ns, na)
        self.q_target_1 = q_target_1.to(self.device)
        self.load_state_dict(self.q_target_1, self.q_net_1.state_dict())

        q_net_2 = QNet(ns, na)
        self.q_net_2 = q_net_2.to(self.device)

        q_target_2 = QNet(ns, na)
        self.q_target_2 = q_target_2.to(self.device)
        self.load_state_dict(self.q_target_2, self.q_net_2.state_dict())

        self.optimizer_q_1 = torch.optim.Adam(self.q_net_1.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                     weight_decay=self.weight_decay_q)

        self.optimizer_q_2 = torch.optim.Adam(self.q_net_2.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                     weight_decay=self.weight_decay_q)

        self.optimizer_p = torch.optim.Adam(self.pi_net.parameters(), lr=self.lr_p, betas=(0.9, 0.999),
                                    weight_decay=self.weight_decay_p)

        self.sample = self.actor_rb

    def play(self):

        if self.env_steps >= self.warmup_steps:
            a, (beta, w) = max_reroute(self.env.s, self.pi_net, self.q_net_1, self.rbi_samples,
                                       self.cmin, self.cmax, self.rbi_greed, self.rbi_epsilon)
        else:

            beta = torch.cuda.FloatTensor(1, self.na, self.rbi_samples).normal_()
            w = torch.ones_like(beta)
            a = torch.cuda.FloatTensor(1, self.na).normal_()

        state = self.env(a)
        state['beta'] = beta
        state['w'] = w

        return state

    def train(self):

        results = defaultdict(lambda: defaultdict(list))

        for i, sample in enumerate(self.sample()):
            i += 1

            s, a, r, t, stag, beta, w = [sample[k] for k in ['s', 'a', 'r', 't', 'stag', 'beta', 'w']]

            self.train()

            with torch.no_grad():
                # self.pi_target(stag)
                # pi_tag_1 = self.pi_target.sample(self.rbi_samples)
                # pi_tag_2 = self.pi_target.sample(self.rbi_samples)
                self.pi_net(stag)
                pi_tag_1 = self.pi_net.sample(self.rbi_samples)
                pi_tag_2 = self.pi_net.sample(self.rbi_samples)
                q_target_1 = self.q_target_1(stag, pi_tag_1).mean(dim=0)
                q_target_2 = self.q_target_2(stag, pi_tag_2).mean(dim=0)

            q_target = torch.min(q_target_1, q_target_2)
            # q_target = q_target_1
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

                beta = beta.permute(2, 0, 1)
                w = w.permute(2, 0, 1)

                self.pi_net(s)
                log_pi = self.pi_net.log_prob(beta)

                loss_p = (1 - self.alpha_rbi) * (- log_pi * w).sum(dim=1).mean()

                # kl div with N(μ, 1)
                mu = self.pi_net.params['loc']
                std_1 = torch.ones_like(mu)

                loss_p += self.alpha_rbi * self.pi_net.kl_divergence((mu, std_1), dirction='backward').sum(dim=1).mean()

                self.optimizer_p.zero_grad()
                loss_p.backward()
                if self.clip_p:
                    nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
                self.optimizer_p.step()

                results['scalar']['q_est'].append(float(-loss_p))

                soft_update(self.pi_net, self.pi_target, self.tau)

            results['scalar']['loss_q'].append(float(loss_q))

            soft_update(self.q_net_1, self.q_target_1, self.tau)
            soft_update(self.q_net_2, self.q_target_2, self.tau)

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

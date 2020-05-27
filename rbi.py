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


def max_reroute(s, pi_net, q_net_1, q_net_2, n=100, cmin=0.5, cmax=1.5, greed=0.1, epsilon=0.01):

    pi_net(s)

    with torch.no_grad():
        beta = pi_net.sample(n)

    # _, b, na = beta.shape
    # s = s.unsqueeze(0).repeat_interleave(n, dim=0)
    # a = beta.view(n * b, na)

    with torch.no_grad():
        qa_1 = q_net_1(s, beta)
        qa_2 = q_net_2(s, beta)
        qa = torch.min(qa_1, qa_2).unsqueeze(-1)

    rank = torch.argsort(torch.argsort(qa, dim=0, descending=True), dim=0, descending=False)
    w = cmin * torch.ones_like(beta)
    m = int((1 - cmin) * n / (cmax - cmin))

    w += (cmax - cmin) * (rank < m).float()
    w += ((1 - cmin) * n - m * (cmax - cmin)) * (rank == m).float()

    w -= greed
    w += greed * n * (rank == 0).float()

    w = w * (1 - epsilon) + epsilon

    w = w.permute(1, 2, 0)
    beta = beta.permute(1, 2, 0)

    w = w / w.sum(dim=2, keepdim=True)

    prob = torch.distributions.Categorical(probs=w)

    a = torch.gather(beta, 2, prob.sample().unsqueeze(2)).squeeze(2)

    return a, (beta, w)


def max_kl(s, pi_net, q_net, n=100, lambda_kl=1.):

    pi_net(s)

    with torch.no_grad():
        beta = pi_net.sample(n)

    _, b, na = beta.shape

    s = s.unsqueeze(0).expand(n, *s.shape)
    s = s.view(n * b, *s.shape[2:])
    a = beta.view(n * b, na)

    with torch.no_grad():
        q = q_net(s, a)

    q = q.view(n, b, 1)
    w = torch.softmax((q - q.sum(dim=0, keepdim=True)) / lambda_kl, dim=0)
    w = torch.repeat_interleave(w, na, dim=2)

    w = w.permute(1, 2, 0)
    beta = beta.permute(1, 2, 0)

    w = w / w.sum(dim=2, keepdim=True)

    prob = torch.distributions.Categorical(probs=w)

    a = torch.gather(beta, 2, prob.sample().unsqueeze(2)).squeeze(2)

    return a, (beta, w)


class RBI(Algorithm):

    def __init__(self, *largs, **kwargs):
        super(RBI, self).__init__(*largs, **kwargs)

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

        self.sample = self.actor_rb

    def actor_rb(self):

        self.env_steps = 0
        self.episodes = 0
        tq = tqdm()

        for i in itertools.count():

            self.env.reset()
            self.episodes = i + 1

            while self.env:

                state = self.play()
                self.env_steps += 1
                tq.update(1)

                self.replay_buffer.add(state)

                if not self.env_steps % self.steps_per_train and \
                    (self.replay_buffer.size >= self.min_replay_buffer or \
                     self.replay_buffer.size >= self.replay_buffer_size):

                    for sample in self.replay_buffer.sample(self.consecutive_train, self.batch):
                        yield sample

            # tail = self.env.k
            # for sample in self.replay_buffer.sample(1 * int(tail / self.batch), self.batch, tail=tail):
            #     s, beta, w = [sample[k] for k in ['s', 'beta', 'w']]
            #
            #     beta = beta.permute(2, 0, 1)
            #     w = w.permute(2, 0, 1)
            #
            #     self.pi_net(self.env.s)
            #     log_pi = self.pi_net.log_prob(beta)
            #
            #     loss_p = (1 - self.alpha_rbi) * (- log_pi * w).sum(dim=1).mean()
            #
            #     # kl div with N(μ, 1)
            #     mu = self.pi_net.params['loc']
            #     std_1 = torch.ones_like(mu)
            #
            #     loss_p += self.alpha_rbi * self.pi_net.kl_divergence((mu, std_1), dirction='backward').mean(dim=1).sum()
            #
            #     self.optimizer_p.zero_grad()
            #     loss_p.backward()
            #     if self.clip_p:
            #         nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
            #     self.optimizer_p.step()

            if self.env_steps >= self.total_steps:
                break

    def play(self):

        if self.env_steps >= self.warmup_steps:
            with torch.no_grad():
                a, (beta_org, w_org) = max_reroute(self.env.s, self.pi_net, self.q_net_1, self.q_net_2, self.rbi_samples,
                                           self.cmin, self.cmax, self.rbi_greed, self.rbi_epsilon)

                # a, (beta_org, w_org) = max_kl(self.env.s, self.pi_net, self.q_net_1, self.rbi_samples,
                #                            self.kl_lambda)

            # # online optimization
            # beta = beta_org.permute(2, 0, 1)
            # w = w_org.permute(2, 0, 1)
            #
            # self.pi_net(self.env.s)
            # log_pi = self.pi_net.log_prob(beta)
            #
            # loss_p = (1 - self.alpha_rbi) * (- log_pi * w).mean(dim=1).sum()
            #
            # # kl div with N(μ, 1)
            # mu = self.pi_net.params['loc']
            # std_1 = torch.ones_like(mu)
            #
            # # loss_p += self.alpha_rbi * self.pi_net.kl_divergence((mu, std_1), dirction='backward').sum(dim=-1).mean()
            # loss_p -= self.alpha_rbi * self.pi_net.entropy().sum(dim=-1).mean()
            #
            # self.optimizer_p.zero_grad()
            # loss_p.backward()
            # if self.clip_p:
            #     nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
            # self.optimizer_p.step()

        else:

            beta_org = torch.cuda.FloatTensor(1, self.na, self.rbi_samples).normal_()
            w_org = torch.ones_like(beta_org)
            a = torch.cuda.FloatTensor(1, self.na).normal_()

        state = self.env(a)

        return state

    def offline_training(self, sample, train_results, n):

        s, a, r, t, stag = [sample[k] for k in ['s', 'a', 'r', 't', 'stag']]

        self.train()

        with torch.no_grad():
            self.pi_net(stag)
            pi_tag_1 = self.pi_net.sample(self.rbi_samples)
            pi_tag_2 = self.pi_net.sample(self.rbi_samples)
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

            _, (beta, w) = max_reroute(s, self.pi_net, self.q_net_1, self.q_net_2, self.rbi_samples,
                                       self.cmin, self.cmax, self.rbi_greed, self.rbi_epsilon)

            beta = beta.permute(2, 0, 1)
            w = w.permute(2, 0, 1)

            self.pi_net(s)
            log_pi = self.pi_net.log_prob(beta)

            loss_p = (1 - self.alpha_rbi) * (- log_pi * w).mean(dim=1).sum()

            loss_p -= self.alpha_rbi * self.pi_net.entropy().sum(dim=-1).mean()

            self.optimizer_p.zero_grad()
            loss_p.backward()
            if self.clip_p:
                nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
            self.optimizer_p.step()

            train_results['scalar']['q_est'].append(float(-loss_p))

            # soft_update(self.pi_net, self.pi_target, self.tau)

        train_results['scalar']['loss_q'].append(float(loss_q))

        soft_update(self.q_net_1, self.q_target_1, self.tau)
        soft_update(self.q_net_2, self.q_target_2, self.tau)

        return train_results


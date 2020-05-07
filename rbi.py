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


def max_reroute(s, pi_net, q_net, n=100, cmin=0.5, cmax=1.5, greed=0.1, epsilon=0.01):

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

    rank = torch.argsort(q, dim=0, descending=True)
    w = cmin * torch.ones_like(beta)
    m = int((1 - cmin) * n / (cmax - cmin))

    w[rank[:m]] += (cmax - cmin)
    w[m] += (1 - cmin) * n - m * (cmax - cmin)

    w -= greed
    w[rank[0]] += greed * n

    w = w * (1 - epsilon) + epsilon

    w = w.permute(1, 2, 0)
    beta = beta.permute(1, 2, 0)

    prob = torch.distributions.Categorical(probs=w)

    a = torch.gather(beta, 2, prob.sample().unsqueeze(2)).squeeze(2)
    return a, (beta, w)


class RBI(Algorithm):

    def __init__(self, env):
        super(RBI, self).__init__()

        self.env = env
        n_a = env.env.action_space.shape[0]
        n_s = env.env.observation_space.shape[0]

        pi_net = PiNet(n_s, n_a, distribution='Normal')
        self.pi_net = pi_net.to(self.device)

        pi_target = PiNet(n_s, n_a, distribution='Normal')
        self.pi_target = pi_target.to(self.device)
        self.load_state_dict(self.pi_target, self.pi_net.state_dict())

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
                                     weight_decay=1e-2)

        self.optimizer_q_2 = torch.optim.Adam(self.q_net_2.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                     weight_decay=1e-2)

        # eps = 1e-04,
        self.optimizer_p = torch.optim.Adam(self.pi_net.parameters(), lr=self.lr_p, betas=(0.9, 0.999),
                                    weight_decay=0)

        self.noise = OrnsteinUhlenbeckActionNoise(torch.zeros(1, n_a).to(self.device),
                                                  self.epsilon * torch.ones(1, n_a).to(self.device))

        self.env_steps = 0
        self.episodes = 0
        self.replay_buffer = defaultdict(list)

    def sample(self):

        for i, state in enumerate(self.play()):

            self.env_steps = i + 1
            if self.env_steps >= self.total_steps:
                break

            for k, v in state.items():
                self.replay_buffer[k].append(v)

            if not i % self.steps_per_train and i >= self.min_replay_buffer:

                for k, v in self.replay_buffer.items():
                    self.replay_buffer[k] = v[-self.replay_memory_size-self.n_steps:]

                indices = torch.randint(len(self.replay_buffer['s']) - self.n_steps, size=(self.consecutive_train, self.batch))

                for index in indices:
                    index_0 = itemgetter(*index)
                    index_n = itemgetter(*(index + self.n_steps))

                    sample = {'s': torch.cat(index_0(self.replay_buffer['s'])),
                              'a': torch.cat(index_0(self.replay_buffer['a'])),
                              'stag': torch.cat(index_n(self.replay_buffer['s'])),
                              'r': torch.cat(index_0(self.replay_buffer['r'])),
                              't': torch.cat(index_0(self.replay_buffer['t'])),
                              'beta': torch.cat(index_0(self.replay_buffer['beta'])),
                               'w': torch.cat(index_0(self.replay_buffer['w']))
                              }
                    yield sample

    def play(self):

        self.env_steps = 0
        self.noise.reset()
        self.eval()

        for i in itertools.count():

            self.episodes = i + 1
            self.env.reset()
            self.noise.reset()

            while self.env:

                noise = self.noise()
                with torch.no_grad():

                    a, (beta, w) = max_reroute(self.env.s, self.pi_net, self.q_net_1, self.rbi_samples,
                                               self.cmin, self.cmax, self.rbi_greed, self.rbi_epsilon)

                a = torch.clamp(a + noise, min=-1, max=1)

                state = self.env(a)
                state['beta'] = beta
                state['w'] = w
                # state = self.env(a)
                yield state
                self.eval()

            # print(f'episode {self.episodes}\t| score {self.env.score}\t| length {self.env.k}')

    def train(self):

        results = defaultdict(lambda: defaultdict(list))

        for i, sample in tqdm(enumerate(self.sample())):
            i += 1

            s, a, r, t, stag, beta, w = [sample[k] for k in ['s', 'a', 'r', 't', 'stag', 'beta', 'w']]

            self.train()

            with torch.no_grad():
                self.pi_target(stag)
                pi_tag = self.pi_target.sample(self.rbi_samples)
                q_target_1 = self.q_target_1(stag, pi_tag)
                q_target_2 = self.q_target_2(stag, pi_tag)

            q_target = torch.min(q_target_1, q_target_2)
            g = r + (1 - t) * self.gamma ** self.n_steps * (q_target * pi_tag).sum(dim=0)

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

                loss_p = (- log_pi * w).sum(dim=0)

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

            # if not n % self.target_update:
            #     self.load_state_dict(self.pi_target, self.pi_net.state_dict())
            #     self.load_state_dict(self.q_target, self.q_net.state_dict())

            if not i % self.train_epoch:

                statistics = self.env.get_stats()
                for k, v in statistics.items():
                    for ki, vi in v.items():
                        results[k][ki] = vi

                results['scalar']['rb'] = len(self.replay_buffer['s'])
                results['scalar']['env-steps'] = self.env_steps
                results['scalar']['episodes'] = self.episodes
                results['scalar']['train-steps'] = i

                yield results
                results = defaultdict(lambda: defaultdict(list))

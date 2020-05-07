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

# Sample = namedtuple('Sample', ('s', 'a', 'r', 't', 'stag', 'pi'))


class RBI(Algorithm):

    def __init__(self, env):
        super(RBI, self).__init__()

        self.env = env
        n_a = env.env.action_space.shape[0]
        n_s = env.env.observation_space.shape[0]

        pi_net = PiNet(n_s, n_a, method='gaussian')
        self.pi_net = pi_net.to(self.device)

        pi_target = PiNet(n_s, n_a, method='gaussian')
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

            for k in state.keys():
                v = getattr(state, k)
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
                              'pi': (torch.cat(index_0(self.replay_buffer['pi_mu'])),
                                     torch.cat(index_0(self.replay_buffer['pi_std'])))
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

                    pi_mu, pi_std = self.pi_net(self.env.s) + noise

                    # if self.env_steps >= self.warmup_steps:
                    #
                    #     # a = self.pi_net(self.env.s, noise=noise)
                    # else:
                    #     a = noise

                state = self.env(torch.clamp(a, min=-1, max=1))
                state['pi_mu'] = pi_mu
                state['pi_std'] = pi_std
                # state = self.env(a)
                yield state
                self.eval()

            # print(f'episode {self.episodes}\t| score {self.env.score}\t| length {self.env.k}')

    def train(self):

        results = defaultdict(lambda: defaultdict(list))

        for i, sample in tqdm(enumerate(self.sample())):
            i += 1

            s, a, r, t, stag, pi = sample['s'], sample['a'], sample['r'], sample['t'], sample['stag'], sample['pi']

            self.train()

            with torch.no_grad():
                pi_tag = self.pi_target(stag)
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

            if not i % self.delayed_policy_update:

                pi_hat, (mu, std) = self.pi_net(s)

                loss_p = self.pi_net.kl_divergence(*pi)

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

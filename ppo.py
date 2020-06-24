from model import MultipleOptimizer, QNet, PiNet
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
from utils import soft_update, OrnsteinUhlenbeckActionNoise, RandomNoise, generalized_advantage_estimation


class SACQ(Algorithm):

    def __init__(self, *largs, **kwargs):
        super(SACQ, self).__init__(*largs, **kwargs)

        self.pi_net = PiNet(self.ns, self.na, distribution='Normal').to(self.device)
        self.v_net = QNet(self.ns, 1).to(self.device)

        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                              weight_decay=self.weight_decay_q)

        self.optimizer_p = torch.optim.Adam(self.pi_net.parameters(), lr=self.lr_p, betas=(0.9, 0.999),
                                    weight_decay=self.weight_decay_p)

    def play(self, env, evaluate=False):

        if self.env_steps >= self.warmup_steps or evaluate:
            with torch.no_grad():
                a = self.pi_net(env.s, evaluate=evaluate)
        else:
            a = None

        state = env(a)
        return state

    def episodic_training(self, train_results, k):

        episode = self.replay_buffer.get_tail(k)

        sl = episode['s']
        sl = torch.chunk(sl, int(len(sl) + 1))
        sl[-1] = torch.cat([sl[-1], episode['stag'][-1].unsqueeze(0)])

        s, a, r, t, stag, e = [episode[k] for k in ['s', 'a', 'r', 't', 'stag', 'e']]

        v = []
        for s in sl:
            v.append(self.v_net(s))

        v = torch.cat(v).detach()
        v1, v2 = v[:-1], v[1:]

        a, v_target = generalized_advantage_estimation(r, t, e, v1, v2, self.gamma, self.lambda_gae)



        return train_results

    def replay_buffer_training(self, sample, train_results, n):

        s, a, r, t, stag = [sample[k] for k in ['s', 'a', 'r', 't', 'stag']]

        with torch.no_grad():
            pi_tag = self.pi_net(stag)
            log_pi_tag = self.pi_net.log_prob(pi_tag).sum(dim=1)
            q_target_1 = self.q_target_1(stag, pi_tag)
            q_target_2 = self.q_target_2(stag, pi_tag)

            q_target = torch.min(q_target_1, q_target_2) - self.alpha * log_pi_tag
            # q_target = torch.min(q_target_1, q_target_2) + self.alpha * self.pi_net.entropy().sum(dim=1)
            g = r + (1 - t) * self.gamma ** self.n_steps * q_target

        qa = self.q_net_1(s, a)
        loss_q_1 = F.mse_loss(qa, g, reduction='mean')

        qa = self.q_net_2(s, a)
        loss_q_2 = F.mse_loss(qa, g, reduction='mean')

        pi = self.pi_net(s)
        qa_1 = self.q_net_1(s, pi)
        qa_2 = self.q_net_2(s, pi)
        qa = torch.min(qa_1, qa_2)
        log_pi = self.pi_net.log_prob(pi).sum(dim=1)

        loss_p = (self.alpha * log_pi - qa).mean()
        # loss_p = (-self.alpha * self.pi_net.entropy().sum(dim=1) - qa).mean()

        with torch.no_grad():
            entropy = self.pi_net.entropy().sum(dim=-1).mean()

        self.optimizer_p.zero_grad()
        loss_p.backward()

        if self.clip_p:
            nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
        self.optimizer_p.step()

        # alpha loss
        if self.entropy_tunning:

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            # alpha_loss = -(self.log_alpha * (-self.pi_net.entropy().sum(dim=1) + self.target_entropy).detach()).mean()

            self.optimizer_alpha.zero_grad()
            alpha_loss.backward()
            self.optimizer_alpha.step()

            self.alpha = float(self.log_alpha.exp())

        train_results['scalar']['alpha'].append(float(self.alpha))
        train_results['scalar']['objective'].append(float(-loss_p))
        train_results['scalar']['entropy'].append(float(entropy))

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


        train_results['scalar']['loss_q_1'].append(float(loss_q_1))
        train_results['scalar']['loss_q_2'].append(float(loss_q_2))

        soft_update(self.q_net_1, self.q_target_1, self.tau)
        soft_update(self.q_net_2, self.q_target_2, self.tau)

        return train_results

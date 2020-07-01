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
from utils import soft_update, OrnsteinUhlenbeckActionNoise, RandomNoise, generalized_advantage_estimation, iter_dict


class PPO(Algorithm):

    def __init__(self, *largs, **kwargs):
        super(PPO, self).__init__(*largs, **kwargs)

        self.pi_net = PiNet(self.ns, self.na, distribution='Normal', bounded=False).to(self.device)
        self.v_net = QNet(self.ns, 0).to(self.device)

        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                              weight_decay=self.weight_decay_q)

        self.optimizer_p = torch.optim.Adam(self.pi_net.parameters(), lr=self.lr_p, betas=(0.9, 0.999),
                                    weight_decay=self.weight_decay_p)

    def play(self, env, evaluate=False):

        with torch.no_grad():
            a = self.pi_net(env.s, evaluate=evaluate)

        if not (self.env_steps >= self.warmup_steps or evaluate):
            a = None

        state = env(a)
        state['logp'] = self.pi_net.log_prob(state['a']).detach()

        return state

    def episodic_training(self, train_results, tail):

        episode = self.replay_buffer.get_tail(tail)

        sl = episode['s']
        sl = list(torch.chunk(sl, int((len(sl) / self.batch) + 1)))
        sl[-1] = torch.cat([sl[-1], episode['stag'][-1].unsqueeze(0)])

        s, r, t, e = [episode[k] for k in ['s', 'r', 't', 'e']]

        v = []
        for s in sl:
            v.append(self.v_net(s))

        v = torch.cat(v).detach()
        v1, v2 = v[:-1], v[1:]

        adv, v_target = generalized_advantage_estimation(r, t, e, v1, v2, self.gamma, self.lambda_gae)

        episode['adv'] = adv
        episode['v_target'] = v_target

        n = self.steps_per_episode * self.batch
        indices = torch.randperm(tail * max(1, n // tail + 1)) % tail
        indices = indices[:n].unsqueeze(1).view(self.steps_per_episode, self.batch)

        samples = {k: v[indices] for k, v in episode.items()}

        for i, sample in enumerate(tqdm(iter_dict(samples))):
            s, a, r, t, stag, adv, v_target, log_pi_old = [sample[k] for k in ['s', 'a', 'r', 't',
                                                                         'stag', 'adv', 'v_target', 'logp']]
            self.pi_net(s)
            log_pi = self.pi_net.log_prob(a)
            ratio = torch.exp((log_pi - log_pi_old).sum(dim=1))

            clip_adv = torch.clamp(ratio, 1 - self.eps_ppo, 1 + self.eps_ppo) * adv
            loss_p = -(torch.min(ratio * adv, clip_adv)).mean()

            approx_kl = -float((log_pi - log_pi_old).sum(dim=1).mean())
            ent = float(self.pi_net.entropy().sum(dim=1).mean())

            if approx_kl > self.target_kl:
                train_results['scalar']['pi_opt_rounds'].append(i)
                break

            clipped = ratio.gt(1 + self.eps_ppo) | ratio.lt(1 - self.eps_ppo)
            clipfrac = float(torch.as_tensor(clipped, dtype=torch.float32).mean())

            self.optimizer_p.zero_grad()
            loss_p.backward()

            if self.clip_p:
                nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
            self.optimizer_p.step()

            train_results['scalar']['loss_p'].append(float(loss_p))
            train_results['scalar']['approx_kl'].append(approx_kl)
            train_results['scalar']['ent'].append(ent)
            train_results['scalar']['clipfrac'].append(clipfrac)

        # for sample in tqdm(iter_dict(samples)):
        #     s, a, r, t, stag, adv, v_target, log_pi_old = [sample[k] for k in ['s', 'a', 'r', 't',
        #                                                                        'stag', 'adv', 'v_target', 'logp']]

            v = self.v_net(s)
            loss_v = F.mse_loss(v, v_target, reduction='mean')

            self.optimizer_v.zero_grad()
            loss_v.backward()
            if self.clip_q:
                nn.utils.clip_grad_norm(self.v_net.parameters(), self.clip_q)
            self.optimizer_v.step()

            train_results['scalar']['loss_v'].append(float(loss_v))

        return train_results


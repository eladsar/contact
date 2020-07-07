from model import MultipleOptimizer, QNet, PiNet
import torch
from torch import nn
import torch.nn.functional as F
from alg import Algorithm
from collections import defaultdict
from tqdm import tqdm
from collections import namedtuple
from utils import soft_update, OrnsteinUhlenbeckActionNoise, RandomNoise

Sample = namedtuple('Sample', ('s', 'a', 'r', 't', 'stag'))


class DDPG(Algorithm):

    def __init__(self, env):
        super(DDPG, self).__init__()

        pi_net = PiNet(self.ns, self.na)
        self.pi_net = pi_net.to(self.device)

        pi_target = PiNet(self.ns, self.na)
        self.pi_target = pi_target.to(self.device)
        self.load_state_dict(self.pi_target, self.pi_net.state_dict())

        q_net = QNet(self.ns, self.na)
        self.q_net = q_net.to(self.device)

        q_target = QNet(self.ns, self.na)
        self.q_target = q_target.to(self.device)
        self.load_state_dict(self.q_target, self.q_net.state_dict())

        self.optimizer_q = torch.optim.Adam(self.q_net.parameters(), lr=self.lr_q, betas=(0.9, 0.999),
                                     weight_decay=1e-2)

        self.optimizer_p = torch.optim.Adam(self.pi_net.parameters(), lr=self.lr_p, betas=(0.9, 0.999),
                                    weight_decay=0)

        self.noise = OrnsteinUhlenbeckActionNoise(torch.zeros(1, self.na).to(self.device),
                                                  self.epsilon * torch.ones(1, self.na).to(self.device))

    def play(self, env, evaluate=False):

        if env.k == 0:
            self.noise.reset()

        noise = self.noise() if not evaluate else 0

        if self.env_steps >= self.warmup_steps or evaluate:
            with torch.no_grad():
                a = self.pi_net(env.s) + noise
                a = torch.clamp(a, min=-1, max=1)
        else:
            a = None

        state = env(a)
        return state

    def train(self):

        results = defaultdict(lambda: defaultdict(list))

        for i, (s, a, r, t, stag) in tqdm(enumerate(self.sample())):
            i += 1
            self.train_mode()
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

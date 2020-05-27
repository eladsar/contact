from model import QNet, PiNet
import torch
from torch import nn
import torch.nn.functional as F
from alg import Algorithm
from utils import soft_update


class PG(Algorithm):

    def __init__(self, *largs, **kwargs):
        super(PG, self).__init__(*largs, **kwargs)

        pi_net = PiNet(self.ns, self.na, distribution='Normal')
        self.pi_net = pi_net.to(self.device)

        v_net = QNet(self.ns, 1)
        self.v_net = v_net.to(self.device)

        v_target = QNet(self.ns, self.na)
        self.v_target = v_target.to(self.device)
        self.load_state_dict(self.v_target, self.v_net.state_dict())

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

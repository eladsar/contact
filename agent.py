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


class Agent(Algorithm):

    def __init__(self, *argv, **kwargs):
        super(Agent, self).__init__()

        self.device = exp.device

        # policy = S2ANet()
        # self.policy = policy.to(self.device)
        #
        # policy_target = S2ANet()
        # self.policy_target = policy_target.to(self.device)
        # self.policy_target.load_state_dict(self.policy.state_dict())
        #
        # q_net = S2ANet()
        # self.q_net = q_net.to(self.device)
        #
        # q_target = S2ANet()
        # self.q_target = q_target.to(self.device)
        # self.q_target.load_state_dict(self.q_net.state_dict())
        #
        # dataset = HexDataset()
        # sampler = UniversalBatchSampler(len(dataset), self.batch)
        # self.loader = torch.utils.data.DataLoader(dataset, batch_size=None,
        #                                                     sampler=sampler,
        #                                                     num_workers=args.cpu_workers)
        #
        # self.optimizer_q = torch.optim.Adam(self.qnet.parameters(),
        #                                    lr=args.lr_d, weight_decay=args.weight_decay,
        #                                    betas=(0., 0.999), eps=1e-4)
        #
        # self.optimizer_p = torch.optim.Adam(self.policy.parameters(),
        #                                    lr=args.lr_g, weight_decay=args.weight_decay,
        #                                    betas=(0., 0.999), eps=1e-4)

    def eval_supervised(self):

        results = defaultdict(lambda: defaultdict(list))
        self.eval()
        for n, sample in tqdm(enumerate(self.loader['test'])):

            x_real = sample['x'].to(self.device)
            z_fake = self.sample_latent(len(x_real))

            x_fake = self.generator(z_fake).detach()

            y_all = self.discriminator(torch.cat([x_real, x_fake]))
            y_real, y_fake = torch.chunk(y_all.detach(), 2)

            results['scalar']['acc_real'].append(float((y_real > 0).float().mean()))
            results['scalar']['acc_fake'].append(float((y_fake < 0).float().mean()))

            if not (n + 1) % self.test_epoch:

                del x_fake
                del x_real
                del z_fake
                del y_all
                del y_real
                del y_fake

                # calculate the condition number

                results['scalar']['cond'] = {}
                for name, param in self.generator.named_parameters():
                    if 'bias' not in name and len(param.shape) > 1:
                        param = param.view(param.shape[0], -1).data.cpu().numpy()
                        results['scalar']['cond'][f'{name}'] = np.linalg.cond(param)

                torch.cuda.empty_cache()
                yield results
                results = defaultdict(lambda: defaultdict(list))
                self.train()

    def train_supervised(self):

        results = defaultdict(lambda: defaultdict(list))
        self.train()

        for n, sample in enumerate(self.loader):

            a, r, s, t, stag = sample['a'].to(self.device), sample['r'].to(self.device), \
                         sample['s'].to(self.device), sample['t'].to(self.device), sample['stag'].to(self.device)

            q_target = self.q_target(s).detach()
            q = self.q_net(s)
            qa = qa.gather(1, a)
            pi = self.policy(s)

            g = r + self.gamma * (1 - t) * (q_target)
            loss_q = F.smooth_l1_loss()


            if not (n + 1) % self.train_epoch:

                if self.fifo:
                    self.load_state_dict(self.discriminator, self.net_buff.get_net0())

                torch.cuda.empty_cache()
                yield results
                results = defaultdict(lambda: defaultdict(list))
                self.train()

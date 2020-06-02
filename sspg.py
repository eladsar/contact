from model import MultipleOptimizer, QNet, PiNet
from config import args, exp
import torch
from torch import nn
import torch.nn.functional as F
from sampler import UniversalBatchSampler, HexDataset
from alg import Algorithm
from utils import soft_update, OrnsteinUhlenbeckActionNoise, RandomNoise, clipped_gd
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

        if self.entropy_tunning:
            self.target_entropy = -torch.prod(torch.Tensor(self.na).to(self.device)).item()
            self.log_alpha = torch.tensor([0.], requires_grad=True, device=self.device)
            self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=self.lr_q)
            self.alpha = float(self.log_alpha.exp())

    def play(self, env, evaluate=False):

        if self.env_steps >= self.warmup_steps or evaluate:
            with torch.no_grad():
                a = self.pi_net(env.s, evaluate=evaluate)
        else:
            a = None

        state = env(a)
        return state

    def replay_buffer_training(self, sample, train_results, n):

        s, a, r, t, stag = [sample[k] for k in ['s', 'a', 'r', 't', 'stag']]

        self.train_mode()
        self.alpha = 0

        with torch.no_grad():
            self.pi_net(stag)
            pi_tag_1 = self.pi_net.sample(self.rbi_learner_samples)
            pi_tag_2 = self.pi_net.sample(self.rbi_learner_samples)
            q_target_1 = self.q_target_1(stag, pi_tag_1).mean(dim=0)
            q_target_2 = self.q_target_2(stag, pi_tag_2).mean(dim=0)

            log_pi_tag = self.pi_net.log_prob(torch.cat([pi_tag_1, pi_tag_2])).mean(dim=0).sum(dim=1)

            q_target = torch.min(q_target_1, q_target_2) - self.alpha * log_pi_tag
            g = r + (1 - t) * self.gamma ** self.n_steps * q_target

        if not n % self.rbi_delayed_policy_update:

            self.pi_net(s)
            pi = self.pi_net.rsample(self.rbi_learner_samples)

            # KL distance with update step

            beta = autograd.Variable(pi.data, requires_grad=True)

            qa_1 = self.q_net_1(s, beta)
            qa_2 = self.q_net_2(s, beta)
            qa = torch.min(qa_1, qa_2)

            gradients = autograd.grad(outputs=qa, inputs=beta, grad_outputs=torch.ones_like(qa),
                                      create_graph=False, retain_graph=False, only_inputs=True)[0]

            # calculate an alternative for the gradient
            lr = .001
            # beta = (beta + lr * gradients / torch.norm(gradients, dim=-1, keepdim=True)).detach()

            beta = clipped_gd(beta, gradients, lr, 1.).detach()

            log_pi = self.pi_net.log_prob(pi)
            log_beta = self.pi_net.log_prob(beta)

            with torch.no_grad():
                qa_1 = self.q_net_1(s, beta)
                qa_2 = self.q_net_2(s, beta)
                qatag = torch.min(qa_1, qa_2).unsqueeze(-1)

            cmin = 0.5
            cmax = 1.5

            rank = torch.argsort(torch.argsort(qatag, dim=0, descending=True), dim=0, descending=False)
            w = cmin * torch.ones_like(beta)
            m = int((1 - cmin) * n / (cmax - cmin))

            w += (cmax - cmin) * (rank < m).float()
            w += ((1 - cmin) * n - m * (cmax - cmin)) * (rank == m).float()

            # loss_p = (self.alpha * log_pi - log_beta).mean()
            loss_p = - (w * (log_beta - log_pi)).sum(dim=-1).mean(dim=0).sum()

            with torch.no_grad():
                entropy = self.pi_net.entropy().sum(dim=-1).mean()

            # numerical gradient (different score)

            # beta = autograd.Variable(pi.data, requires_grad=True)
            #
            # qa_1 = self.q_net_1(s, beta)
            # qa_2 = self.q_net_2(s, beta)
            # qa = torch.min(qa_1, qa_2)
            #
            # gradients = autograd.grad(outputs=qa, inputs=beta, grad_outputs=torch.ones_like(qa),
            #                           create_graph=False, retain_graph=False, only_inputs=True)[0]
            #
            # # calculate an alternative for the gradient
            # lr = 0.01
            # beta = (beta + lr * gradients).detach()
            #
            # with torch.no_grad():
            #     qa_1 = self.q_net_1(s, beta)
            #     qa_2 = self.q_net_2(s, beta)
            #     # qatag = torch.min(qa_1, qa_2)
            #     qatag = (qa_1 + qa_2) / 2
            #
            # dq = (qatag - qa.detach()) / torch.norm(lr * gradients, dim=-1, keepdim=True)
            # ngrad = gradients / torch.norm(gradients, dim=-1, keepdim=True)
            # gradients = dq.unsqueeze(-1) * ngrad
            #
            #
            # log_pi = self.pi_net.log_prob(pi).sum(dim=-1).mean(dim=0)
            # dq = (pi * gradients.detach()).sum(dim=-1).mean(dim=0)
            #
            # loss_p = (self.alpha * log_pi - dq).mean()
            #
            # with torch.no_grad():
            #     entropy = self.pi_net.entropy().sum(dim=-1).mean()



            # algernative gradient (same score)

            # beta = autograd.Variable(pi.data, requires_grad=True)
            #
            # qa_1 = self.q_net_1(s, beta)
            # qa_2 = self.q_net_2(s, beta)
            # qa = torch.min(qa_1, qa_2)
            #
            # gradients = autograd.grad(outputs=qa, inputs=beta, grad_outputs=torch.ones_like(qa),
            #                           create_graph=False, retain_graph=False, only_inputs=True)[0]
            #
            # log_pi = self.pi_net.log_prob(pi).sum(dim=-1).mean(dim=0)
            # dq = (pi * gradients.detach()).sum(dim=-1).mean(dim=0)
            #
            # loss_p = (self.alpha * log_pi - dq).mean()
            #
            # with torch.no_grad():
            #     entropy = self.pi_net.entropy().sum(dim=-1).mean()


            # ORIGINAL FORMULATION

            # qa_1 = self.q_net_1(s, pi).mean(dim=0)
            # qa_2 = self.q_net_2(s, pi).mean(dim=0)
            # qa = torch.min(qa_1, qa_2)
            #
            # log_pi = self.pi_net.log_prob(pi).mean(dim=0).sum(dim=1)
            #
            # loss_p = (self.alpha * log_pi - qa).mean()
            #
            # with torch.no_grad():
            #     entropy = self.pi_net.entropy().sum(dim=-1).mean()

            # entropy = self.pi_net.entropy().sum(dim=-1).mean()
            # loss_p -= 0 * entropy




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
            # soft_update(self.pi_net, self.pi_target, self.tau)

        qa = self.q_net_1(s, a)
        loss_q_1 = F.mse_loss(qa, g, reduction='mean')

        qa = self.q_net_2(s, a)
        loss_q_2 = F.mse_loss(qa, g, reduction='mean')

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


from model import QNet, PiNet
import torch
from torch import nn
import torch.nn.functional as F
from alg import Algorithm
from utils import soft_update
import torch.autograd as autograd
import math


def max_reroute(s, pi_net, q_net_1, q_net_2, n=100, cmin=0.5, cmax=1.5, greed=0.1, epsilon=0.01, lr=0.01):

    with torch.no_grad():
        pi_net(s)
        beta = pi_net.sample(n)

    beta = autograd.Variable(beta.data, requires_grad=True)
    qa_1 = q_net_1(s, beta)
    qa_2 = q_net_2(s, beta)
    # CHECK THE UNSQUEEZE IN ACTION!!
    qa = torch.min(qa_1, qa_2).unsqueeze(-1)
    gradients = autograd.grad(outputs=qa, inputs=beta, grad_outputs=torch.cuda.FloatTensor(qa.size()).fill_(1.),
                              create_graph=False, retain_graph=False, only_inputs=True)[0]

    beta = (beta + lr * gradients).detach()

    with torch.no_grad():
        qa_1 = q_net_1(s, beta)
        qa_2 = q_net_2(s, beta)
        qatag = torch.min(qa_1, qa_2).unsqueeze(-1)

    rank = torch.argsort(torch.argsort(qatag, dim=0, descending=True), dim=0, descending=False)
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

        self.alpha = self.rbi_alpha
        if self.entropy_tunning:
            # self.target_entropy = -float(self.na)
            std_target = 0.3 / math.sqrt(self.na)
            self.target_entropy = self.na * 0.5 * math.log(2 * math.pi * math.e * (std_target ** 2))
            print(f'target entropy: {self.target_entropy}')
            self.lr_alpha = 0.01
            # self.target_entropy = 1.5

    def online_training(self, state, train_results):

        state.pop('beta')
        state.pop('w')

        return state, train_results

    # def play(self, env, evaluate=False):
    #
    #     if self.env_steps >= self.warmup_steps or evaluate:
    #         with torch.no_grad():
    #             a = self.pi_net(env.s, evaluate=evaluate)
    #     else:
    #         a = None
    #
    #     beta = torch.cuda.FloatTensor(1, self.na, self.rbi_actor_samples).normal_()
    #     w = torch.ones_like(beta)
    #
    #     state = env(a)
    #     state['beta'] = beta
    #     state['w'] = w
    #
    #     return state

    def play(self, env, evaluate=False):

        if evaluate:

            self.pi_net(env.s)
            a = self.pi_net.sample(evaluate=True)
            state = env(a)
            return state

        if self.env_steps >= self.warmup_steps:
            a, (beta, w) = max_reroute(env.s, self.pi_net, self.q_net_1, self.q_net_2, n=self.rbi_actor_samples,
                                       cmin=self.cmin, cmax=self.cmax, greed=self.rbi_greed, epsilon=self.rbi_epsilon,
                                       lr=self.rbi_lr)

            a = self.pi_net.squash(self.pi_net.desquash(a) + self.rbi_alpha * torch.zeros_like(a).normal_())

        else:

            beta = torch.cuda.FloatTensor(1, self.na, self.rbi_actor_samples).normal_()
            w = torch.ones_like(beta)
            a = None

        state = env(a)
        state['beta'] = beta
        state['w'] = w

        return state

    def replay_buffer_training(self, sample, train_results, n):

        s, a, r, t, stag = [sample[k] for k in ['s', 'a', 'r', 't', 'stag']]

        self.train_mode()

        # if n % 2:
        #     q_net, optimizer_q, q_target, tag = self.q_net_1, self.optimizer_q_1, self.q_target_1, '1'
        # else:
        #     q_net, optimizer_q, q_target, tag = self.q_net_2, self.optimizer_q_2, self.q_target_2, '2'

        # with torch.no_grad():
        #     pi_tag = self.pi_net(stag)
        #     log_pi_tag = self.pi_net.log_prob(pi_tag).sum(dim=1)
        #     q_target_1 = self.q_target_1(stag, pi_tag)
        #     q_target_2 = self.q_target_2(stag, pi_tag)
        #
        #     q_target = torch.min(q_target_1, q_target_2) - self.alpha * log_pi_tag
        #     g = r + (1 - t) * self.gamma ** self.n_steps * q_target

        # with torch.no_grad():
        #     pi_tag = self.pi_net(stag)
        #     q_target_1 = self.q_target_1(stag, pi_tag)
        #     q_target_2 = self.q_target_2(stag, pi_tag)
        #
        #     q_target = torch.min(q_target_1, q_target_2)
        #     g = r + (1 - t) * self.gamma ** self.n_steps * q_target

        with torch.no_grad():
            self.pi_net(stag)
            pi_tag_1 = self.pi_net.sample(self.rbi_learner_samples)
            pi_tag_2 = self.pi_net.sample(self.rbi_learner_samples)
            q_target_1 = self.q_target_1(stag, pi_tag_1).mean(dim=0)
            q_target_2 = self.q_target_2(stag, pi_tag_2).mean(dim=0)

            v_target = torch.min(q_target_1, q_target_2)
            g = r + (1 - t) * self.gamma ** self.n_steps * v_target

        if not n % self.rbi_delayed_policy_update:

            _, (beta, w) = max_reroute(s, self.pi_net, self.q_net_1, self.q_net_2, n=self.rbi_learner_samples,
                                       cmin=self.cmin, cmax=self.cmax, greed=self.rbi_greed, epsilon=self.rbi_epsilon,
                                       lr=self.rbi_lr)

            beta = beta.permute(2, 0, 1)
            w = w.permute(2, 0, 1)

            self.pi_net(s)
            log_pi = self.pi_net.log_prob(beta)

            loss_p = (- log_pi * w).mean(dim=1).sum()

            entropy = self.pi_net.entropy().sum(dim=-1).mean()
            # entropy = -(w * self.pi_net.log_prob(beta)).sum(dim=-1).mean()

            loss_p -= self.alpha * entropy

            self.optimizer_p.zero_grad()
            loss_p.backward()
            # train_results['scalar']['grad'].append(float(gret_grad_norm(self.pi_net)))
            if self.clip_p:
                nn.utils.clip_grad_norm(self.pi_net.parameters(), self.clip_p)
            self.optimizer_p.step()

            train_results['scalar']['objective'].append(float(-loss_p))
            train_results['scalar']['entropy'].append(float(entropy))

            # # alpha loss
            if self.entropy_tunning:

                self.alpha = float(self.alpha * (1 - self.lr_alpha) + self.lr_alpha * F.relu(self.target_entropy - entropy))
                train_results['scalar']['alpha'].append(float(self.alpha))

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

        # qa = q_net(s, a)
        # loss_q = F.mse_loss(qa, g, reduction='mean')
        #
        # optimizer_q.zero_grad()
        # loss_q.backward()
        # if self.clip_q:
        #     nn.utils.clip_grad_norm(q_net.parameters(), self.clip_q)
        # optimizer_q.step()
        #
        # train_results['scalar'][f'loss_q_{tag}'].append(float(loss_q))
        #
        # soft_update(q_net, q_target, self.tau)

        return train_results


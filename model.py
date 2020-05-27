from torch import nn
import torch
import torch.nn.functional as F
from config import args, exp
from torch.autograd import Function
from collections import namedtuple
from torchvision import transforms
from torch.nn.utils import spectral_norm
import math
import numpy as np
import itertools


class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        print("Reversed")
        print(grads)
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class GPFuncLayer(nn.Module):

    def __init__(self, func, *argc, **kwargs):
        super(GPFuncLayer, self).__init__()
        self.func = func
        self.argc = argc
        self.kwargs = kwargs

    def forward(self, x):
        return self.func(x, *self.argc, **self.kwargs)


class GPAttrLayer(nn.Module):

    def __init__(self, func, *argc, **kwargs):
        super(GPAttrLayer, self).__init__()
        self.func = func
        self.argc = argc
        self.kwargs = kwargs

    def forward(self, x):

        f = getattr(x, self.func)
        return f(*self.argc, **self.kwargs)


class GlobalBlock(nn.Module):

    def __init__(self, planes):
        super(GlobalBlock, self).__init__()

        self.query = nn.Sequential(
            #             nn.BatchNorm1d(planes, affine=True),
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True)
            #             spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
        )

        self.key = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True)
        )

        self.value = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True)
        )

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True)
        )

        self.planes = planes

    def forward(self, x):
        q = self.query(x).transpose(1, 2)
        k = self.key(x)
        v = self.value(x).transpose(1, 2)

        a = torch.softmax(torch.bmm(q, k) / math.sqrt(self.planes), dim=2)
        r = torch.bmm(a, v).transpose(1, 2)
        r = self.output(r)

        return x + r


class LinearHead(nn.Module):

    def __init__(self, in_features, out_classes):
        super(LinearHead, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_classes))
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        y = torch.mm(x, self.weight)
        w = torch.norm(self.weight, dim=0, keepdim=True)

        x = torch.norm(x, dim=1, keepdim=True)
        y = y / (x * w)

        # optional:
        # y = torch.tan(math.pi / 2 * y)

        if y.shape[-1] == 1:
            y = y.squeeze(-1)

        return y


# class FCNet(nn.Module):
#
#     def __init__(self, n=1):
#         super(FCNet, self).__init__()
#         #         self.fc = nn.Sequential(nn.Linear(actions, layer, bias=True),
#         #                                 activation(),
#         #                                 nn.Linear(layer, layer, bias=True),
#         #                                 activation(),
#         #                                 nn.Linear(layer, layer//2, bias=True),
#         #                                 activation(),
#         #                                 nn.Linear(layer//2, n, bias=True))
#
#         self.fc = nn.Sequential(nn.Linear(actions, layer, bias=True),
#                                 SeqResBlock(n_res, layer),
#                                 nn.ReLU(),
#                                 nn.Linear(layer, n, bias=True))
#
#         self.n = n
#
#         init_weights(self, init='ortho')
#
#     def forward(self, x):
#         x = self.fc(x)
#         if self.n == 1:
#             x = x.squeeze(1)
#
#         return x


class SplineEmbedding(nn.Module):

    def __init__(self, actions, emb=32, delta=10):
        super(SplineEmbedding, self).__init__()

        self.delta = delta
        self.actions = actions
        self.emb = emb

        self.register_buffer('ind_offset', torch.arange(self.actions, dtype=torch.int64).unsqueeze(0))

        self.b = nn.Embedding((2 * self.delta + 1) * actions, emb, sparse=True)

    def forward(self, x):
        n = len(x)

        xl = (x * self.delta).floor()
        xli = self.actions * (xl.long() + self.delta) + self.ind_offset
        xl = xl / self.delta
        xli = xli.view(-1)

        xh = (x * self.delta + 1).floor()
        xhi = self.actions * (xh.long() + self.delta) + self.ind_offset
        xh = xh / self.delta
        xhi = xhi.view(-1)

        bl = self.b(xli).view(n, self.actions, self.emb)
        bh = self.b(xhi).view(n, self.actions, self.emb)

        delta = 1 / self.delta

        x = x.unsqueeze(2)
        xl = xl.unsqueeze(2)
        xh = xh.unsqueeze(2)

        h = bh / delta * (x - xl) + bl / delta * (xh - x)

        return h


class SeqResBlock(nn.Module):

    def __init__(self, n_res, layer):
        super(SeqResBlock, self).__init__()

        self.seq_res = nn.ModuleList([ResBlock(layer) for _ in range(n_res)])

    def forward(self, x):
        for res in self.seq_res:
            x = res(x)

        return x


class ResBlock(nn.Module):

    def __init__(self, layer):
        super(ResBlock, self).__init__()

        self.fc = nn.Sequential(nn.ReLU(),
                                nn.Linear(layer, layer, bias=True),
                                nn.ReLU(),
                                nn.Linear(layer, layer, bias=True),
                                )

    def forward(self, x):
        h = self.fc(x)
        return x + h


class GlobalModule(nn.Module):

    def __init__(self, planes):
        super(GlobalModule, self).__init__()

        self.blocks = nn.Sequential(
            #             GlobalBlock(planes),
            GlobalBlock(planes),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = x.squeeze(2)

        return x


class SplineHead(nn.Module):

    def __init__(self, actions, layer, n=1, emb=32, emb2=32, n_res=2):
        super(SplineHead, self).__init__()

        self.emb = emb
        self.actions = actions
        self.n = n
        self.emb2 = emb2

        #         parallel = 1

        self.global_interaction = GlobalModule(emb)

        input_len = emb + actions

        #         input_len = (emb + 1) * actions

        self.fc = nn.Sequential(nn.Linear(input_len, layer, bias=True),
                                SeqResBlock(n_res, layer),
                                nn.ReLU(),
                                nn.Linear(layer, n, bias=True))

        #         self.fc = nn.Sequential(nn.Linear(parallel * emb + actions, layer, bias=True),
        # #                                 nn.ReLU(),
        # #                                 nn.Linear(layer, layer, bias=True),
        # #                                 nn.ReLU(),
        # #                                 nn.Linear(layer, layer, bias=True),
        # #                                 nn.ReLU(),
        #                         nn.Linear(layer, layer, bias=True),
        #                         nn.ReLU(),
        #                         nn.Linear(layer, layer//2, bias=True),
        #                         nn.ReLU(),
        #                         nn.Linear(layer//2, n, bias=True))

        init_weights(self, init=args.init)

    def forward(self, x, x_emb):
        h = x_emb.transpose(2, 1)
        h = self.global_interaction(h)

        x = torch.cat([x, h], dim=1)

        x = self.fc(x)
        if self.n == 1:
            x = x.squeeze(1)

        return x


class SplineNet(nn.Module):

    def __init__(self, actions, layer, n=1, emb=32, emb2=32, n_res=2):
        super(SplineNet, self).__init__()

        self.embedding = SplineEmbedding(actions, emb=emb)
        self.head = SplineHead(actions, layer, n=n, emb=emb, emb2=emb2, n_res=n_res)

    def forward(self, x):
        x_emb = self.embedding(x)
        x = self.head(x, x_emb)

        return x


class QSpline(nn.Module):

    def __init__(self, states, actions, layer=128, hidden=128, emb=32, emb2=32, n_res=2):
        super(QSpline, self).__init__()

        self.s_net = SplineNet(states, layer=layer, n=hidden, emb=emb, emb2=emb2, n_res=n_res)
        self.a_net = SplineNet(actions, layer=layer, n=hidden, emb=emb, emb2=emb2, n_res=n_res)

        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden),
                                  nn.ReLU(),
                                  nn.Linear(hidden, 1),
                                 )

    def forward(self, s, a):
        s = self.s_net(s)
        a = self.a_net(a)

        x = self.head(torch.cat([s, a], dim=1))

        return x


class PiSpline(nn.Module):

    def __init__(self, states, actions, layer=128, emb=32, emb2=32, n_res=2):
        super(PiSpline, self).__init__()

        self.s_net = SplineNet(states, layer=layer, n=actions, emb=emb, emb2=emb2, n_res=n_res)

    def forward(self, s):
        x = self.s_net(s)
        return x


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


def tanh_grad(x):
    x = torch.clamp_min(1 - torch.tanh(x) ** 2, min=1e-8)
    return torch.log(x)


def atanh(x):
    x = torch.clamp(x, min=-1+1e-5, max=1-1e-5)
    return 0.5 * torch.log((1 + x) / (1 - x))


class Deterministic(torch.distributions.distribution.Distribution):

    def __init__(self, loc):
        super(Deterministic, self).__init__()
        self.x = loc

    def rsample(self, sample_shape=torch.Size([])):

        expanded = sample_shape + self.x.shape

        x = self.x.view(torch.Size([1] * len(sample_shape)) + self.x.shape)
        x = x.expand(expanded)

        return x

    def log_prob(self, value):
        return torch.ones_like(self.x)

    def entropy(self):
        return torch.zeros_like(self.x)


def identity(x):
    return x


def zero(x):
    return torch.zeros_like(x)


class Policy(nn.Module):

    def __init__(self, distribution, bounded=True):
        super(Policy, self).__init__()

        if bounded:
            self.squash = torch.tanh
            self.desquash = atanh
            self.da_du = tanh_grad

        else:
            self.squash = identity
            self.desquash = identity
            self.da_du = zero

        if distribution == 'deterministic':
            self.generator = Deterministic
        else:
            self.generator = getattr(torch.distributions, distribution)

        self.distribution = None

    def kl_divergence(self, args, dirction='forward'):

        q = self.generator(*args)
        if dirction == 'forward':
            return torch.distributions.kl_divergence(self.distribution, q)
        else:
            return torch.distributions.kl_divergence(q, self.distribution)

    def _sample(self, method, n=None, evaluate=False):

        if evaluate:
            sample = self.distribution.mean
            if type(n) is int:
                sample = torch.repeat_interleave(sample.unsqueeze(0), n, dim=0)

            if method == 'sample':
                sample = sample.detach()

        elif n is None:
            sample = getattr(self.distribution, method)()
        else:

            if type(n) is int:
                n = torch.Size([n])
            sample = getattr(self.distribution, method)(n)

        a = self.squash(sample)
        return a

    def rsample(self, n=None, evaluate=False):
        return self._sample('rsample', n=n, evaluate=evaluate)

    def sample(self, n=None, evaluate=False):
        return self._sample('sample', n=n, evaluate=evaluate)

    def log_prob(self, a):

        distribution = self.distribution.expand(a.shape)

        sample = self.desquash(a)

        # log_prob = distribution.log_prob(sample) - self.da_du(sample).sum(dim=-1, keepdim=True)
        log_prob = distribution.log_prob(sample) - self.da_du(sample)

        return log_prob

    def entropy(self):

        return self.distribution.entropy()

    def forward(self, evaluate=False, **params):

        self.params = params

        self.distribution = self.generator(**params)
        a = self.rsample(evaluate=evaluate)

        return a.squeeze(0)


class MedianNorm(nn.Module):

    def __init__(self):
        super(MedianNorm, self).__init__()

    def forward(self, x):

        with torch.no_grad():
            mu = torch.median(x)

        x = x - mu

        return x


class ActorTD3(Policy):
    def __init__(self, state_dim, action_dim, distribution='deterministic', bounded=True):
        super(ActorTD3, self).__init__(distribution, bounded=bounded)

        # self.l1 = nn.Linear(state_dim, 256, bias=args.bias_p)
        # self.l2 = nn.Linear(256, 256, bias=args.bias_p)

        self.lin = nn.Sequential(nn.Linear(state_dim, 256, bias=args.bias_p),
                                 # nn.LayerNorm(256, elementwise_affine=False),
                                 # MedianNorm(),
                                 nn.ReLU(),
                                 nn.Linear(256, 256, bias=args.bias_p),
                                 # nn.LayerNorm(256, elementwise_affine=False),
                                 # MedianNorm(),
                                 nn.ReLU(),
                                 )

        self.mu_head = nn.Linear(256, action_dim, bias=args.bias_p)

        self.form = distribution

        if distribution in ['Normal', 'Uniform']:
            self.std_head = nn.Linear(256, action_dim)

    def forward(self, s, evaluate=False):

        s = self.lin(s)

        mu = self.mu_head(s)

        if self.form in ['Normal', 'Uniform']:
            logstd = self.std_head(s)
            logstd = torch.clamp(logstd, min=args.min_log, max=args.max_log)
            std = logstd.exp()
            params = {'loc': mu, 'scale': std}
        else:
            params = {'loc': mu}

        a = super(ActorTD3, self).forward(**params, evaluate=evaluate)
        return a


# class ActorMoG(Policy):
#     def __init__(self, state_dim, action_dim, mixtures, bounded=True):
#         super(ActorMoG, self).__init__('Normal', bounded=bounded)
#
#         self.l1 = nn.Linear(state_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#
#         self.mu_head = nn.Linear(256, action_dim * mixtures)
#         self.attention = nn.Linear(256, action_dim * mixtures)
#         self.std_head = nn.Linear(256, action_dim * mixtures)
#
#         self.na = action_dim
#         self.nm = mixtures
#         self.w = None
#
#     def entropy(self):
#
#         return self.distribution.entropy()
#
#     def _sample(self, method, n=None, evaluate=False):
#
#         prob = torch.distributions.Categorical(probs=self.w)
#
#         if evaluate:
#             sample = self.distribution.mean
#             if type(n) is int:
#                 sample = torch.repeat_interleave(sample.unsqueeze(0), n, dim=0)
#
#             if method == 'sample':
#                 sample = sample.detach()
#
#         elif n is None:
#             sample = getattr(self.distribution, method)()
#             prob = prob.sample()
#         else:
#
#             if type(n) is int:
#                 n = torch.Size([n])
#             sample = getattr(self.distribution, method)(n)
#             prob = prob.sample(n)
#
#         a = self.squash(sample)
#
#         a = a.gather(a, 2, prob.unsqueeze(2)).squeeze(2)
#
#         return a
#
#     def log_prob(self, a):
#
#         distribution = self.distribution.expand(a.shape)
#         w = self.w.expand(a.shape)
#         sample = self.desquash(a)
#
#         # log_prob = distribution.log_prob(sample) - self.da_du(sample).sum(dim=-1, keepdim=True)
#         log_prob = distribution.log_prob(sample) - self.da_du(sample) + torch.log(w)
#
#         return log_prob
#
#     def forward(self, s):
#         s = F.relu(self.l1(s))
#         s = F.relu(self.l2(s))
#         mu = self.mu_head(s)
#         mu = mu.view(len(mu), self.na, self.nm)
#         w = self.attention(s)
#         self.w = torch.softmax(w, dim=-1)
#
#         logstd = self.std_head(s)
#         logstd = torch.clamp(logstd, min=args.min_log, max=args.max_log)
#         std = logstd.exp()
#         params = {'loc': mu, 'scale': std}
#
#         a = super(ActorMoG, self).forward(**params)
#         return a


class CriticTD3(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticTD3, self).__init__()

        self.actions = action_dim
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, s, a):

        shape = s.shape
        if self.actions:
            if len(a.shape) > len(shape):
                n, b, _ = shape = a.shape

                s = s.unsqueeze(0).expand(n, b, -1)

                s = torch.cat([s, a], dim=-1)
                s = s.view(n * b, -1)
            else:
                s = torch.cat([s, a], dim=-1)

        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        q = self.l3(s)

        q = q.view(*shape[:-1], -1).squeeze(-1)

        return q


# QNet = QSpline
# PiNet = PiSpline

# QNet = QVanilla
# PiNet = PiVanilla

# QNet = Critic
# PiNet = Actor

QNet = CriticTD3
PiNet = ActorTD3

#
# class S2ANet(nn.Module):
#
#     def __init__(self):
#         super(S2ANet, self).__init__()
#
#         self.actions = args.board ** 2
#
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=True),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),)
#
#         self.attention = nn.Sequential(GlobalBlock(128),
#                                        nn.AdaptiveAvgPool1d(1),
#                                        nn.Flatten(),
#                                        )
#         self.head = LinearHead(128, self.actions)
#         init_weights(self)
#
#     def forward(self, s):
#
#         s = self.cnn(s)
#         s = self.attention(s)
#         a = self.head(s)
#
#         return a


def init_weights(net, init='ortho'):
    net.param_count = 0
    for module in net.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear,
                               nn.ConvTranspose2d, nn.ConvTranspose1d)):
            if init == 'ortho':
                torch.nn.init.orthogonal_(module.weight)
            elif init == 'N02':
                torch.nn.init.normal_(module.weight, 0, 0.02)
            elif init in ['glorot', 'xavier']:
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                print('Init style not recognized...')
        net.param_count += sum([p.data.nelement() for p in module.parameters()])

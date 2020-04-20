from torch import nn
import torch
import torch.nn.functional as F
from config import args, exp
from torch.autograd import Function
from collections import namedtuple
from torchvision import transforms
from torch.nn.utils import spectral_norm
import math


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


class FCNet(nn.Module):

    def __init__(self, n=1):
        super(FCNet, self).__init__()
        #         self.fc = nn.Sequential(nn.Linear(actions, layer, bias=True),
        #                                 activation(),
        #                                 nn.Linear(layer, layer, bias=True),
        #                                 activation(),
        #                                 nn.Linear(layer, layer//2, bias=True),
        #                                 activation(),
        #                                 nn.Linear(layer//2, n, bias=True))

        self.fc = nn.Sequential(nn.Linear(actions, layer, bias=True),
                                SeqResBlock(n_res, layer),
                                nn.ReLU(),
                                nn.Linear(layer, n, bias=True))

        self.n = n

        init_weights(self, init='ortho')

    def forward(self, x):
        x = self.fc(x)
        if self.n == 1:
            x = x.squeeze(1)

        return x


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


class QVanilla(nn.Module):

    def __init__(self, states, actions, hidden_s=256, hidden_a=32):
        super(QVanilla, self).__init__()

        self.s_net = nn.Sequential(nn.Linear(states, hidden_s),
                                   nn.ReLU(),)

        self.a_net = nn.Sequential(nn.Linear(actions, hidden_a),
                                   nn.ReLU(),)

        self.head = nn.Sequential(nn.Linear(hidden_s + hidden_a, hidden_s),
                                  nn.ReLU(),
                                  nn.Linear(hidden_s, 1),
                                  )

    def forward(self, s, a):
        s = self.s_net(s)
        a = self.a_net(a)

        x = self.head(torch.cat([s, a], dim=1))

        return x.squeeze(1)


class PiVanilla(nn.Module):

    def __init__(self, states, actions, layer=128):
        super(PiVanilla, self).__init__()

        self.s_net = nn.Sequential(nn.Linear(states, layer),
                                   nn.ReLU(),
                                   nn.Linear(layer, actions),)

    def forward(self, s):
        a = self.s_net(s)
        a = torch.tanh(a)

        return a


# QNet = QSpline
# PiNet = PiSpline

QNet = QVanilla
PiNet = PiVanilla


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

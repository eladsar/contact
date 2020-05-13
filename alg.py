from config import args, exp
import torch
from torch import nn
import torch.nn.functional as F
import copy
from collections import defaultdict
from apex import amp
from loguru import logger
import warnings
import itertools
from sampler import ReplayBuffer
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)


class Algorithm(object):

    def __init__(self):
        self.networks_dict = {}
        self.optimizers_dict = {}

        self.multi_gpu = False
        self.device = exp.device
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.env = None
        self.env_steps = 0
        self.episodes = 0

    def postprocess(self, sample):

        for name, var in sample.items():
            sample[name] = var.to(self.device)

        return sample

    def reset_opt(self, optimizer):

        optimizer.state = defaultdict(dict)

    def reset_networks(self, networks_dict, optimizers_dict):

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        for net in networks_dict:
            net = getattr(self, net)
            net.apply(init_weights)

        for optim in optimizers_dict:
            optim = getattr(self, optim)
            optim.state = defaultdict(dict)

    def get_optimizers(self):

        self.optimizers_dict = {}

        for d in dir(self):
            x = getattr(self, d)
            if issubclass(type(x), torch.optim.Optimizer) and hasattr(x, 'state_dict'):
                self.optimizers_dict[d] = x

        return self.optimizers_dict

    def get_networks(self):

        self.networks_dict = {}
        name_dict = {}

        for d in dir(self):
            x = getattr(self, d)
            if issubclass(type(x), nn.Module) and hasattr(x, 'state_dict'):
                if next(x.parameters(), None) is not None:
                    name_dict[d] = getattr(x, 'named_parameters')
                    self.networks_dict[d] = x

        return name_dict

    def play(self):
        raise NotImplementedError

    def actor_rb(self):

        self.env_steps = 0
        self.episodes = 0
        tq = tqdm()

        for i in itertools.count():

            self.env.reset()
            self.episodes = i + 1

            while self.env:

                state = self.play()
                self.env_steps += 1
                tq.update(1)

                self.replay_buffer.add(state)

                if not self.env_steps % self.steps_per_train and \
                    (self.replay_buffer.size >= self.min_replay_buffer or \
                     self.replay_buffer.size >= self.replay_buffer_size):

                    for sample in self.replay_buffer.sample(self.consecutive_train, self.batch):
                        yield sample

            if self.env_steps >= self.total_steps:
                break

    def train(self):

        if not self.networks_dict:
            self.get_networks()

        for net in self.networks_dict.values():
            net.train()

    def eval(self):

        if not self.networks_dict:
            self.get_networks()

        for net in self.networks_dict.values():
            net.eval()

    def state_dict(self, net):

        if self.multi_gpu:
            return copy.deepcopy(net.module.state_dict())
        return copy.deepcopy(net.state_dict())

    def load_state_dict(self, net, state):

        if self.multi_gpu:
            net.module.load_state_dict(state, strict=False)
        else:
            net.load_state_dict(state, strict=False)

    def store_net_0(self):

        if not self.networks_dict:
            self.get_networks()

        self.net_0 = {}

        for name, net in self.networks_dict.items():
            net.eval()

            self.net_0[name] = self.state_dict(net)

    def save_checkpoint(self, path=None, aux=None, save_model=False):

        if not self.networks_dict:
            self.get_networks()
        if not self.optimizers_dict:
            self.get_optimizers()

        state = {'aux': aux}
        try:
            state['amp'] = amp.state_dict()
        except:
            pass

        for net in self.networks_dict:
            state[net] = self.state_dict(self.networks_dict[net])
            if save_model:
                state[f"{net}_model"] = self.networks_dict[net]

        for optimizer in self.optimizers_dict:
            state[optimizer] = copy.deepcopy(self.optimizers_dict[optimizer].state_dict())

        if path is not None:
            torch.save(state, path)

        return state

    def load_checkpoint(self, pathstate):

        if not self.networks_dict:
            self.get_networks()
            self.get_optimizers()

        if type(pathstate) is str:
            state = torch.load(pathstate, map_location="cuda:%d" % args.cuda)
        else:
            state = pathstate

        for net in self.networks_dict:
            self.load_state_dict(self.networks_dict[net], state[net])

        for optimizer in self.optimizers_dict:
            self.optimizers_dict[optimizer].load_state_dict(state[optimizer])
            # pass

        try:
            amp.load_state_dict(state['amp'])
        except Exception as e:
            logger.error(str(e))

        return state['aux']




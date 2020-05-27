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
import math

warnings.filterwarnings('ignore', category=UserWarning)


class Algorithm(object):

    def __init__(self, *largs, **kwargs):

        self.env_train = kwargs['env_train']
        self.env_eval = kwargs['env_eval']

        self.na = self.env_train.action_space.shape[0]
        self.ns = self.env_train.observation_space.shape[0]

        self.networks_dict = {}
        self.optimizers_dict = {}

        self.multi_gpu = False
        self.device = exp.device
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.env_steps = 0
        self.episodes = 0
        self.epsilon /= math.sqrt(self.na)
        self.alpha_rbi /= math.sqrt(self.na)
        self.rbi_lr /= math.sqrt(self.na)

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

    def play(self, env, evaluate=False):
        raise NotImplementedError

    def step(self):

        self.env_steps = 0
        self.episodes = 0

        for i in itertools.count():

            self.env_train.reset()
            self.episodes = i + 1

            while self.env_train:

                state = self.play(self.env_train)
                self.env_steps += 1
                yield state

    def replay_buffer_training(self, sample, train_results, n):
        return train_results

    def online_training(self, state, train_results):
        return state, train_results

    def episodic_training(self, state, train_results):
        return train_results

    def train_mode(self):

        if not self.networks_dict:
            self.get_networks()

        for net in self.networks_dict.values():
            net.train()

    def eval_mode(self):

        if not self.networks_dict:
            self.get_networks()

        for net in self.networks_dict.values():
            net.eval()

    def reinforcement_training(self):

        self.replay_buffer.reset()
        train_results = defaultdict(lambda: defaultdict(list))
        test_results = defaultdict(lambda: defaultdict(list))

        for i, state in tqdm(enumerate(self.step())):
            i += 1

            state, train_results = self.online_training(state, train_results)
            self.replay_buffer.add(state)

            if not self.env_steps % self.steps_per_train and (self.replay_buffer.size >= self.min_replay_buffer or
               self.replay_buffer.size >= self.replay_buffer_size):

                for j, sample in enumerate(self.replay_buffer.sample(self.consecutive_train, self.batch)):

                    n = i * self.consecutive_train + j
                    train_results = self.replay_buffer_training(sample, train_results, n)

                if not self.env_train:
                    train_results = self.episodic_training(state, train_results)

            if not i % self.test_epoch:
                test_results = self.eval(test_results, i)

            if not i % self.train_epoch:

                statistics = self.env_train.get_stats()
                for k, v in statistics.items():
                    for ki, vi in v.items():
                        train_results[k][ki] = vi

                train_results['scalar']['rb'] = self.replay_buffer.size
                train_results['scalar']['env-steps'] = self.env_steps
                train_results['scalar']['episodes'] = self.episodes
                train_results['scalar']['train-steps'] = i

                yield train_results, test_results
                train_results = defaultdict(lambda: defaultdict(list))
                test_results = defaultdict(lambda: defaultdict(list))

            if i >= self.total_steps:
                break

    def eval(self, test_results, n):

        for i in tqdm(range(self.test_episodes)):

            self.env_eval.reset()

            while self.env_eval:
                self.play(self.env_eval, evaluate=True)

            test_results['scalar']['score'].append(self.env_eval.score)
            test_results['scalar']['length'].append(self.env_eval.k)

        test_results['scalar']['n'] = n
        return test_results

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




import numpy as np
import torch
import pybullet_envs
import gym
from collections import namedtuple, defaultdict
from utils import Identity, ZFilter, RewardFilter

# State = namedtuple('State', ('s', 'a', 'r', 't', 'k', 'stag'))


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        """
        Normalizes the actions to be in between action_space.high and action_space.low.
        If action_space.low == -action_space.high, this is equals to action_space.high*action.

        :param action:
        :return: normalized action
        """
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def reverse_action(self, action):
        """
        Reverts the normalization

        :param action:
        :return:
        """
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


class Environment(object):

    def __init__(self, *args, **kwargs):
        self.image = None
        self.s, self.r, self.t = None, None, False
        self.score = 0
        self.k = 0  # Internal step counter

        self.statistics = defaultdict(lambda: defaultdict(list))

    def get_stats(self):

        statistics = self.statistics
        self.statistics = defaultdict(lambda: defaultdict(list))
        return statistics

    def reset(self):
        raise NotImplementedError

    def step(self, a):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def __bool__(self):
        return not bool(self.t)


def squash(a):
    return np.clip(a, a_max=1, a_min=-1)


def desquash(a):
    return a


# def squash(a):
#     return np.clip(np.tanh(a), a_min=-1+1e-6, a_max=1-1e-6)
#
#
# def desquash(a):
#     return np.arctanh(a)


class BulletEnv(Environment):

    def __init__(self, name, render=False, cuda=True, n_steps=1, gamma=0.99, render_mode='rgb_array',
                 norm_rewards='none', norm_states=False, clip_obs=0, clip_rew=0):

        super(BulletEnv, self).__init__()

        self.torch = torch.cuda if cuda else torch
        self.name = name
        self.render_mode = render_mode
        self.gamma = gamma
        # self.env = NormalizedActions(gym.make(name, render=render))
        # self.env = gym.make(name, render=render)
        self.env = gym.make(name)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.ns = self.observation_space.shape[0]

        # Support for state normalization or using time as a feature
        self.state_filter = Identity()
        if norm_states:
            self.state_filter = ZFilter(self.state_filter, shape=[self.ns], clip=clip_obs)
        # Support for rewards normalization
        self.reward_filter = Identity()
        if norm_rewards == "rewards":
            self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=clip_rew)
        elif norm_rewards == "returns":
            self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=self.gamma, clip=clip_rew)

        self.n_steps = n_steps
        self.gamma = gamma ** (n_steps - self.torch.FloatTensor(n_steps).fill_(1).cumsum(0))
        self.render = render
        self.rewards = list(self.torch.FloatTensor(n_steps, 1).zero_())
        self.terminals = list(self.torch.FloatTensor(n_steps, 1).zero_())
        self.reset()

    def process_reward(self, r):

        r = self.reward_filter(r)

        return self.torch.FloatTensor([r])

    def process_state(self, s, reset=False):

        # l = np.nan_to_num(self.env.observation_space.low, nan=0.0, posinf=0.0, neginf=0.0)
        # h = np.nan_to_num(self.env.observation_space.high, nan=0.0, posinf=0.0, neginf=0.0)
        #
        # mu = (h + l) / 2
        # sig = (h - l) / 2
        # sig = sig * (sig > 0) + (sig == 0)
        #
        # s = (s - mu) / sig
        # # s = np.tanh(s)
        s = self.state_filter(s, reset=reset)

        return self.torch.FloatTensor(s).unsqueeze(0)

    def reset(self):

        self.score = 0
        self.k = 0
        self.t = False

        self.state_filter.reset()
        s = self.env.reset()
        self.s = self.process_state(s, reset=True)
        if self.render:
            self.image = self.env.render(mode=self.render_mode)

    def process_action(self, a):

        # assume a torch tensor
        a = a.detach().squeeze(0).cpu().numpy()

        # a_squash = squash(a)
        # bounded_l = self.env.action_space.bounded_below
        # bounded_h = self.env.action_space.bounded_above
        #
        # l = np.nan_to_num(self.env.action_space.low, nan=0.0, posinf=0.0, neginf=0.0)
        # h = np.nan_to_num(self.env.action_space.high, nan=0.0, posinf=0.0, neginf=0.0)
        #
        # mu = (h + l) / 2
        # sig = (h - l) / 2
        # sig = sig * (sig > 0) + (sig == 0)
        # a_squash = sig * a_squash + mu
        #
        # a_scale = sig * a + mu
        # # a_atanh = desquash(a)
        #
        # a = ((a >= 0) * bounded_h + (a < 0) * bounded_l) * a_squash + \
        #     ((a >= 0) * (~bounded_h) + (a < 0) * (~bounded_l)) * a_scale

        return a

    def step(self, a=None):

        if a is None:
            a = self.torch.FloatTensor(self.action_space.sample()).unsqueeze(0)
        if len(a.shape) != 2:
            a = a.unsqueeze(0)

        # Process state
        a_real = self.process_action(a)
        s, r, t, _ = self.env.step(a_real)
        self.k += 1

        self.t = t
        actor_t = t if self.k < self.env._max_episode_steps else False
        self.score += r

        if t:

            self.statistics['scalar']['score'].append(self.score)
            self.statistics['scalar']['length'].append(float(self.k))
            self.statistics['scalar']['avg_r'].append(self.score / float(self.k))

        if self.render:
            self.image = self.env.render(mode=self.render_mode)

        s = self.process_state(s)
        r = self.process_reward(r)

        state = {'s': self.s, 'r': r, 't': self.torch.FloatTensor([int(actor_t)]),
                 'k': self.torch.LongTensor([self.k]), 'a': a, 'stag': s, 'e': self.torch.FloatTensor([not bool(self)])}

        self.s = s

        return state

    # def step(self, a):
    #
    #     # Process state
    #     a_real = self.process_action(a)
    #     s, r, t, _ = self.env.step(a_real)
    #     self.t = t
    #     self.k += 1
    #
    #     self.score += r
    #
    #     terminals = torch.cat(self.terminals)
    #     if terminals.sum() == len(terminals):
    #         list(self.torch.FloatTensor(self.n_steps, 1).zero_())
    #         terminals = torch.cat(self.terminals)
    #
    #     r_traj = ((1 - terminals) * self.gamma * torch.cat(self.rewards)).sum(0, keepdims=True)
    #
    #     if t:
    #         self.statistics['scalar']['score'].append(self.score)
    #         self.statistics['scalar']['length'].append(float(self.k))
    #         self.statistics['scalar']['avg_r'].append(self.score / float(self.k))
    #
    #         self.terminals += [self.torch.FloatTensor([int(t)])] * self.n_steps
    #         self.terminals.pop(0)
    #     else:
    #         if len(self.terminals) == self.n_steps:
    #             self.terminals.append(self.torch.FloatTensor([int(t)]))
    #         self.terminals.pop(0)
    #
    #     self.rewards.append(self.process_reward(r))
    #     self.rewards.pop(0)
    #
    #     t_traj = self.terminals[0].clone()
    #
    #     if self.render:
    #         self.image = self.env.render()
    #
    #     state = State(s=self.s, r=r_traj, t=t_traj,
    #                   k=self.torch.LongTensor([self.k]), a=a)
    #     self.s = self.process_state(s)
    #
    #     return state

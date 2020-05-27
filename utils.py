import torch
import numpy as np


def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# def soft_update(net, target, tau):
#
#     state = net.state_dict()
#
#     with torch.no_grad():
#         for n, p in target.named_parameters():
#             p.data.mul_(1 - tau).add_(tau, state[n].data)


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# and adapted to be synchronous with https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OUNoise:
    def __init__(self, dt=0.01, mu=0, theta=0.15, sigma=0.2):
        self.dt = dt
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones_like(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state

    def __call__(self, *args, **kwargs):
        return self.noise(*args, **kwargs)


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = torch.tensor(dt).float()
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * torch.sqrt(self.dt) * torch.zeros_like(self.mu).normal_()
        self.x_prev = x
        return x

    def __call__(self, *args, **kwargs):
        return self.noise(*args, **kwargs)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class RandomNoise:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def noise(self):
        x = self.mu + self.sigma * torch.zeros_like(self.mu).normal_()
        return x

    def __call__(self, *args, **kwargs):
        return self.noise(*args, **kwargs)

    def reset(self):
        pass




# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


def gret_grad_norm(net, norm_type=2):

    return torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in net.parameters()]), norm_type)


class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    # def push(self, x):
    #     x = np.asarray(x)
    #     assert x.shape == self._M.shape
    #     self._n += 1
    #     if self._n == 1:
    #         self._M[...] = x
    #     else:
    #         oldM = self._M.copy()
    #         self._M[...] = oldM + (x - oldM) / self._n
    #         self._S[...] = self._S + (x - oldM) * (x - self._M)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape


class RewardFilter:
    """
    Incorrect reward normalization [copied from OAI code]
    update return
    divide reward by std(return) without subtracting and adding back mean
    """

    def __init__(self, prev_filter, shape, gamma, clip=None):
        assert shape is not None
        self.gamma = gamma
        self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        # x = x / (self.rs.std + 1e-8)

        x = (x - self.rs.mean) / (self.rs.std + 1e-8)

        if self.clip > 0:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.ret = np.zeros_like(self.ret)
        self.prev_filter.reset()


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, prev_filter, shape, center=True, scale=True, clip=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)
        self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip > 0:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.prev_filter.reset()


class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass


def generalized_advantage_estimation(r, v, gamma, lambda_gae):

    return (1 - lambda_gae)



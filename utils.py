import torch
import numpy as np
from os.path import isdir, join
from fnmatch import fnmatch, filter

def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def clipped_gd(x, dx, eta, gamma):

    norm = torch.norm(dx, dim=-1, keepdim=True)
    eta = torch.clamp_max(gamma * eta / norm, eta)

    return x - eta * dx



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


# def generalized_advantage_estimation(r, t, e, v1, v2, gamma, lambda_gae, norm=False):
#
#     device = r.device
#
#     rewards = []
#     discounted_reward = 0
#     for reward, is_terminal in zip(r.flip(0), reversed(e.flip(0))):
#         if is_terminal:
#             discounted_reward = 0
#         discounted_reward = reward + (gamma * discounted_reward)
#         rewards.insert(0, discounted_reward)
#
#     # Normalizing the rewards:
#     rewards = torch.tensor(rewards).to(device)
#     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
#
#     advantages = rewards - v1.detach()
#
#     a = advantages
#     v = rewards
#
#     return a, v


mu = 0
std = 1
lr = 0.1


def norm_r(r, e, gamma):

    global mu
    global std

    mu_target = r.mean()
    std_target = (r.std() + 1e-3) * max(1 - gamma, 1e-3, float(e.sum() / len(e))) * 10

    mu = (1 - lr) * mu + lr * mu_target
    std = (1 - lr) * std + lr * std_target

    return (r - mu) / std


def generalized_advantage_estimation(r, t, e, v1, v2, gamma, lambda_gae, norm=False):

    device = r.device
    b = torch.FloatTensor([1, 0]).to(device)
    aa = torch.FloatTensor([1, -gamma * lambda_gae]).to(device)
    av = torch.FloatTensor([1, -gamma]).to(device)

    if norm:
        r = norm_r(r, e, gamma)

    i = torch.nonzero(e).flatten()
    if len(i):
        if (i[-1] + 1) == len(e):
            i = torch.cat([torch.LongTensor([0]).to(device), i+1])
        else:
            i = torch.cat([torch.LongTensor([0]).to(device), i + 1, torch.LongTensor([len(e)]).to(device)])

        i = list(i[1:] - i[:-1])

        r = torch.split(r, i)
        t = torch.split(t, i)
        v1 = torch.split(v1, i)
        v2 = torch.split(v2, i)
    else:
        r = [r]
        t = [t]
        v1 = [v1]
        v2 = [v2]

    a = []
    v = []

    for ri, ti, v1i, v2i in zip(r, t, v1, v2):

        delta = -v1i + ri + gamma * (1 - ti) * v2i

        a.append(lfilter(delta.flip(0), aa, b).flip(0))
        v.append(lfilter(ri.flip(0), av, b).flip(0))

    a = torch.cat(a)
    v = torch.cat(v)

    return a, v


def lfilter(waveform, a_coeffs, b_coeffs):
    r"""Perform an IIR filter by evaluating difference equation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`.  Must be normalized to -1 to 1.
        a_coeffs (Tensor): denominator coefficients of difference equation of dimension of `(n_order + 1)`.
                                Lower delays coefficients are first, e.g. `[a0, a1, a2, ...]`.
                                Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (Tensor): numerator coefficients of difference equation of dimension of `(n_order + 1)`.
                                 Lower delays coefficients are first, e.g. `[b0, b1, b2, ...]`.
                                 Must be same size as a_coeffs (pad with 0's as necessary).

    Returns:
        Tensor: Waveform with dimension of `(..., time)`.  Output will be clipped to -1 to 1.
    """
    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    assert (a_coeffs.size(0) == b_coeffs.size(0))
    assert (len(waveform.size()) == 2)
    assert (waveform.device == a_coeffs.device)
    assert (b_coeffs.device == a_coeffs.device)

    device = waveform.device
    dtype = waveform.dtype
    n_channel, n_sample = waveform.size()
    n_order = a_coeffs.size(0)
    n_sample_padded = n_sample + n_order - 1
    assert (n_order > 0)

    # Pad the input and create output
    padded_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)
    padded_waveform[:, (n_order - 1):] = waveform
    padded_output_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)

    # Set up the coefficients matrix
    # Flip coefficients' order
    a_coeffs_flipped = a_coeffs.flip(0)
    b_coeffs_flipped = b_coeffs.flip(0)

    # calculate windowed_input_signal in parallel
    # create indices of original with shape (n_channel, n_order, n_sample)
    window_idxs = torch.arange(n_sample, device=device).unsqueeze(0) + torch.arange(n_order, device=device).unsqueeze(1)
    window_idxs = window_idxs.repeat(n_channel, 1, 1)
    window_idxs += (torch.arange(n_channel, device=device).unsqueeze(-1).unsqueeze(-1) * n_sample_padded)
    window_idxs = window_idxs.long()
    # (n_order, ) matmul (n_channel, n_order, n_sample) -> (n_channel, n_sample)
    input_signal_windows = torch.matmul(b_coeffs_flipped, torch.take(padded_waveform, window_idxs))

    for i_sample, o0 in enumerate(input_signal_windows.t()):
        windowed_output_signal = padded_output_waveform[:, i_sample:(i_sample + n_order)]
        o0.sub_(torch.mv(windowed_output_signal, a_coeffs_flipped))
        o0.div_(a_coeffs[0])

        padded_output_waveform[:, i_sample + n_order - 1] = o0

    # output = torch.clamp(padded_output_waveform[:, (n_order - 1):], min=-1., max=1.)
    output = padded_output_waveform[:, (n_order - 1):]

    # unpack batch
    output = output.view(shape[:-1] + output.shape[-1:])

    return output


def iter_dict(d):
    for i in range(len(d[list(d.keys())[0]])):
        yield {k: v[i] for k, v in d.items()}


def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not isdir(join(path, name)))
        return ignore
    return _ignore_patterns
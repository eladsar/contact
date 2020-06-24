import torch.utils.data
import numpy as np
import torch
import os
from config import args, exp, set_seed
import torchvision
from torchvision import transforms
import itertools
import pandas as pd


class HexDataset(torch.utils.data.Dataset):

    def __init__(self):
        super(HexDataset, self).__init__()

        self.dataset = pd.read_pickle(exp.dataset_dir)

        self.tensor = {'a': lambda x: torch.LongTensor(np.stack(x)).unsqueeze(1),
                       'r': torch.FloatTensor,
                       's': lambda x: torch.FloatTensor(np.stack(x)),
                       'i': torch.LongTensor,
                       't': torch.FloatTensor}

    def __getitem__(self, index):
        return {k: self.tensor[k](v.values) for k, v in self.dataset.iloc[index].items()}

    def __len__(self):
        return len(self.dataset)


class UniversalBatchSampler(object):

    def __init__(self, size, batch, epochs=None):

        self.length = np.inf if epochs is None else epochs
        self.batch = batch
        self.size = size
        self.minibatches = int(self.size / self.batch)

    def __iter__(self):

        for i in itertools.count():
            if i >= self.length:
                break

            shuffle_indexes = np.random.choice(np.arange(self.size), (self.minibatches, self.batch), replace=False)

            for i in range(self.minibatches):
                samples = shuffle_indexes[i]
                yield samples

    def __len__(self):
        return self.length * self.minibatches


class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.states = {}

    def reset(self):
        self.ptr = 0
        self.size = 0
        self.states = {}

    def add(self, state):

        if not len(self.states):
            for k, v in state.items():
                self.states[k] = torch.repeat_interleave(torch.zeros_like(v), self.max_size, dim=0)

        for k in self.states.keys():
            self.states[k][self.ptr] = state[k]

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_tail(self, tail):
        return {k: v[:-tail] for k, v in self.states.items()}

    def sample(self, consecutive_train, batch_size, tail=None):

        if tail is None:
            indices = torch.randint(self.size, size=(consecutive_train, batch_size))
        else:
            tail = min(tail, self.size)
            indices = torch.randint(tail, size=(consecutive_train, batch_size))
            indices = (indices - self.ptr) % self.size

        for ind in indices:
            yield {k: v[ind] for k, v in self.states.items()}


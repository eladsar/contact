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


import json
import os
from copy import deepcopy
from glob import glob
from random import shuffle

import chainer
import numpy as np
from chainer.dataset import DatasetMixin


def list_converter(batch, device=None, padding=None):
    """retrun lists of xp.array
    """
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    assert isinstance(first_elem, tuple)
    result = []
    if not isinstance(padding, tuple):
        padding = [padding] * len(first_elem)

    for i in range(len(first_elem)):
        result.append([chainer.Variable(chainer.dataset.convert.to_device(
            device, example[i])) for example in batch])

    return tuple(result)


class NPZDataset(DatasetMixin):

    def __init__(self, dataset_root, dataset_dir="", param_file="datasetparam.json"):
        if dataset_dir:
            data_dir = os.path.join(dataset_root, dataset_dir)
        else:
            data_dir = dataset_root
        paths = sorted(
            glob(os.path.join(data_dir, '**/*.npz'), recursive=True))
        self._paths = paths

        try:
            with open(os.path.join(dataset_root, param_file), 'r') as f:
                load = json.load(f)
            self.params = load

        except FileNotFoundError:
           self.params = None 

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = self._paths[i]
        return dict(np.load(path))

    def get_example_from_names(self, names, random=True):
        names = deepcopy(names)
        if random:
            shuffle(names)

        path = None
        for name in names:
            for p in self._paths:
                if name == os.path.basename(p):
                    print(p)
                    path = p
                    break

            if path is not None:
                break

        if path is None:
            data = path = None
        else:
            data = dict(np.load(path))

        return data, path

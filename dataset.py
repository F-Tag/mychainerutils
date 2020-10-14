
import json
import os
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from random import shuffle

import numpy as np
import six
from chainer.dataset import DatasetMixin, to_device
from chainer.dataset.convert import _concat_arrays


class NPZDataset(DatasetMixin):

    def __init__(self, dataset_root, include_key=None, exclude_key=None, label_dct=None, param_file="datasetparam.json"):
        dataset_root = Path(dataset_root).expanduser()

        self.label_level = len(dataset_root.parts)

        if include_key is not None:
            include_key = '|'.join(include_key)
            include_checker = re.compile(include_key)

        if exclude_key is not None:
            exclude_key = '|'.join(exclude_key)
            exclude_checker = re.compile(exclude_key)

        dirs = []
        for d in sorted(dataset_root.glob('*')):
            if not os.path.isdir(d):
                continue

            if include_key is not None:
                if include_checker.search(os.path.basename(d)) is not None:
                    dirs.append(d)
                    continue
            else:
                if exclude_key is not None:
                    if exclude_checker.search(os.path.basename(d)) is None:
                        dirs.append(d)

                else:
                    dirs.append(d)

        paths = []
        labels = []
        index = 0
        if label_dct is None:
            label_dct = {}
        for d in dirs:
            tmp = sorted(list(d.glob('**/*.npz')))

            if len(tmp) == 0:
                continue

            key = os.path.basename(d)
            if key not in label_dct:
                values = list(label_dct.values())
                while True:
                    if index in values:
                        index += 1
                    else:
                        label_dct[key] = index
                        break

            paths += tmp
            labels += [label_dct[key]] * len(tmp)

        assert len(paths) == len(labels)

        self._paths = paths
        self.label_dct = label_dct

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
        tmp = dict(np.load(path, mmap_mode="r"))
        tmp["label_name"] = path.parts[self.label_level]
        tmp['label'] = self.label_dct[tmp["label_name"]]
        return tmp

    def get_example_from_names(self, names, random=True):
        names = deepcopy(names)
        if random:
            shuffle(names)

        path = None
        for name in names:
            for i, p in enumerate(self._paths):
                if name == os.path.basename(p):
                    path = p
                    idx = i
                    break

            if path is not None:
                break

        if path is None:
            data = path = None
        else:
            data = dict(np.load(path))
            data['labels'] = self.label_dct[path.parts[self.label_level]]

        return data, path


class PathDataset(DatasetMixin):

    def __init__(self, ext, dataset_root, include_key=None, exclude_key=None, label_dct=None):
        dataset_root = Path(dataset_root).expanduser()

        self.label_level = len(dataset_root.parts)

        if include_key is not None:
            include_key = '|'.join(include_key)
            include_checker = re.compile(include_key)

        if exclude_key is not None:
            exclude_key = '|'.join(exclude_key)
            exclude_checker = re.compile(exclude_key)

        dirs = []
        for d in sorted(dataset_root.glob('*')):
            if not os.path.isdir(d):
                continue
            if exclude_key is not None:
                if exclude_checker.search(os.path.basename(d)) is not None:
                    continue
            if include_key is not None:
                if include_checker.search(os.path.basename(d)) is None:
                    continue
            dirs.append(d)

        paths = []
        labels = []
        index = 0
        if label_dct is None:
            label_dct = {}
        for d in dirs:
            tmp = sorted(list(d.glob('**/*' + ext)))

            if len(tmp) == 0:
                continue

            key = os.path.basename(d)
            if key not in label_dct:
                values = list(label_dct.values())
                while True:
                    if index in values:
                        index += 1
                    else:
                        label_dct[key] = index
                        break

            paths += tmp
            labels += [label_dct[key]] * len(tmp)

        assert len(paths) == len(labels)

        self._paths = paths
        self.label_dct = label_dct

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = self._paths[i]
        label_name = path.parts[self.label_level]
        label = self.label_dct[label_name]
        return path, label, label_name

    def get_example_from_names(self, names, random=True):
        names = deepcopy(names)
        if random:
            shuffle(names)

        path = None
        for name in names:
            for i, p in enumerate(self._paths):
                if name == os.path.basename(p):
                    path = p
                    idx = i
                    break

            if path is not None:
                break

        if path is None:
            return None, None

        else:
            return (path, self.label_dct[path.parts[self.label_level]]), path


def list_examples(batch, device=None, padding=None):

    if len(batch) == 0:
        raise ValueError('batch is empty')

    if padding is not None:
        raise NotImplementedError

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            result.append([to_device(device, example[i]) for example in batch])

        return tuple(result)

    elif isinstance(first_elem, dict):
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}

        for key in first_elem:
            result[key] = [to_device(device, example[key])
                           for example in batch]

        return result

    else:
        raise NotImplementedError
        return to_device(device, _concat_arrays(batch, padding))

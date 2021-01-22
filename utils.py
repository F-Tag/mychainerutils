
from io import BytesIO, TextIOWrapper
from os import environ
from pathlib import Path
from zipfile import ZipFile

import chainer
import chainer.functions as F
import numpy as np
from chainer import dataset


def get_shuffled_example(length):
    origin = np.arange(length)
    shuffled = np.random.permutation(origin)
    dup_index = np.where(origin == shuffled)[0].tolist()
    dup = len((dup_index))
    if dup == 1:
        while True:
            idx = np.random.randint(length)
            if idx != dup_index[0]:
                break
        shuffled[dup_index + [idx]] = shuffled[[idx] + dup_index]
    elif dup == 2:
        shuffled[dup_index] = shuffled[dup_index][::-1]
    elif dup >= 3:
        ch_s_index = get_shuffled_example(dup)
        shuffled[dup_index] = shuffled[dup_index][ch_s_index]

    return shuffled


def get_datasetroot():
    if "DATASET_ROOT" in environ:
        ret = Path(environ["DATASET_ROOT"])
    else:
        ret = Path("~", "dataset")

    ret = ret.expanduser()

    ret.mkdir(exist_ok=True, parents=True)

    return ret


def get_saveroot():
    if "SAVE_ROOT" in environ:
        ret = Path(environ["SAVE_ROOT"])
    else:
        ret = Path("./results")

    ret = ret.expanduser()

    ret.mkdir(exist_ok=True, parents=True)

    return ret


def to_numpy(v):
    return F.copy(chainer.as_variable(v), -1).array


def get_loss_scale():
    if chainer.config.dtype == chainer.mixed16:
        ret = "dynamic"
    else:
        ret = None

    return ret


def del_link_hooks(chain, name="WeightNormalization"):
    for link in chain.children():
        if name in link.local_link_hooks:
            link.delete_hook(name)
        del_link_hooks(link, name)


def set_config(func, name, value):
    def ret(*args, **kwargs):
        with chainer.using_config(name, value):
            return func(*args, **kwargs)
    return ret


def load_modelzipfile(url, param_name="hparams.json", model_name="model_last", root="~/.models"):

    tmp = dataset.get_dataset_root()
    dataset.set_dataset_root(Path(root).expanduser())
    path = dataset.cached_download(url)
    dataset.set_dataset_root(tmp)

    with ZipFile(path) as f:
        namelist = [name for name in f.namelist() if name[-1] !=
                    "/" if name[-1] != "/"]

        param = [name for name in namelist if param_name == Path(name).name]
        assert len(param) == 1
        param = param.pop()
        with TextIOWrapper(f.open(param, "r"), encoding="utf-8") as pf:
            param = pf.read()

        model = [name for name in namelist if model_name == Path(name).name]
        assert len(model) == 1
        model = model.pop()
        model = BytesIO(f.read(model))

    return param, model


from functools import partial
from inspect import signature
from math import ceil
from warnings import warn

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import configuration
from chainer import functions as F
from chainer import variable
from chainer.backends import cuda

from . import functions as mF


class TSRegressor(L.Classifier):
    compute_accuracy = False

    def __init__(self, predictor,
                 lossfun=F.absolute_error, label_key=-1, return_key=None, delta=True, deltadelta=True):

        try:
            signature(lossfun).parameters['reduce']
        except KeyError:
            pass
        else:
            lossfun = partial(lossfun, reduce='no')

        super().__init__(predictor,
                         lossfun=F.absolute_error,
                         label_key=label_key)
        self.return_key = return_key
        self.delta = True
        self.deltadelta = True

    def __call__(self, *args, **kwargs):

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None
        batch = len(t)
        y = self.predictor(*args, **kwargs)
        if self.return_key is not None and type(y) == tuple:
            self.y = y[self.return_key]
        else:
            self.y = y
        diff = self.lossfun(F.concat(self.y, axis=0), F.concat(t, axis=0))
        self.loss = F.sum(diff) / batch
        chainer.reporter.report({'loss': self.loss}, self)

        if self.delta:
            arr1 = [arr[:-2] for arr in self.y]
            arr2 = [arr[2:] for arr in self.y]
            arr_y = 0.5 * F.concat(arr2, axis=0) - 0.5 * F.concat(arr1, axis=0)

            arr1 = [arr[:-2] for arr in t]
            arr2 = [arr[2:] for arr in t]
            arr_t = 0.5 * F.concat(arr2, axis=0) - 0.5 * F.concat(arr1, axis=0)

            loss = F.sum(self.lossfun(arr_y, arr_t)) / batch
            chainer.reporter.report({'delta': loss}, self)
            self.loss += loss

        if self.deltadelta:
            arr1 = [arr[:-2] for arr in self.y]
            arr2 = [arr[2:] for arr in self.y]
            arr3 = [arr[1:-1] for arr in self.y]
            arr_y = F.concat(arr2, axis=0) + F.concat(arr1,
                                                      axis=0) - 2 * F.concat(arr3, axis=0)

            arr1 = [arr[:-2] for arr in t]
            arr2 = [arr[2:] for arr in t]
            arr3 = [arr[1:-1] for arr in t]
            arr_t = F.concat(arr2, axis=0) + F.concat(arr1,
                                                      axis=0) - 2 * F.concat(arr3, axis=0)

            loss = F.sum(self.lossfun(arr_y, arr_t)) / batch
            chainer.reporter.report({'deltadelta': loss}, self)
            self.loss += loss

        return self.loss


class Regressor(L.Classifier):
    compute_accuracy = False

    def __init__(self, predictor,
                 lossfun=F.mean_absolute_error, label_key=-1, return_key=None):

        super().__init__(predictor,
                         lossfun=F.mean_absolute_error,
                         label_key=label_key)
        self.return_key = return_key

    def __call__(self, *args, **kwargs):

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None
        y = self.predictor(*args, **kwargs)
        if self.return_key is not None and type(y) == tuple:
            self.y = y[self.return_key]
        else:
            self.y = y
        self.loss = self.lossfun(self.y, t)
        chainer.reporter.report({'loss': self.loss}, self)
        return self.loss


class HighWayLayers(chainer.Chain):
    """
    Nstep HighWay
    """

    def __init__(self, in_out_size, n_layers, nobias=False, activate='relu', init_Wh=None, init_Wt=None, init_bh=None, init_bt=-1):
        layers = chainer.ChainList()
        [layers.add_link(L.Highway(in_out_size, nobias, mF.get_function(
            activate), init_Wh, init_Wt, init_bh, init_bt)) for _ in range(n_layers)]
        super().__init__()
        with self.init_scope():
            self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class ConvBN1D(chainer.Chain):
    """
    conv1D + BN
    """

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, initialW=None, initial_bias=None):
        self.finetune = False
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution1D(
                in_channels, out_channels, ksize, stride, pad, True, initialW, initial_bias)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        return self.bn(self.conv(x), finetune=self.finetune)


class HighWayConv1D(L.Convolution1D):
    def __init__(self, in_out_channels, ksize, stride=1, pad='center', nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1):
        if pad == 'causal':
            pad = (ksize - 1) * dilate
        elif pad == 'center':
            pad = ceil((ksize - 1) * dilate / 2)
        else:
            raise ValueError

        super().__init__(in_out_channels, in_out_channels * 2, ksize, stride,
                         pad, nobias, initialW, initial_bias, dilate=dilate, groups=groups)

    def __call__(self, x):
        length = x.shape[-1]
        h1 = super().__call__(x)[..., :length]
        h2, h3 = F.split_axis(h1, 2, 1)
        h4 = F.sigmoid(h2)
        return h4 * h3 + (1 - h4) * x


def build_mlp(n_out, n_units=256, layers=5, normalize=None, activation='leaky_relu', dropout=0.0):
    net = chainer.Sequential()
    if normalize == 'BN' or normalize == 'LN':
        nobias = True
    else:
        nobias = False

    for _ in range(layers):
        net.append(L.Linear(n_units, nobias=nobias))
        if normalize == 'BN':
            net.append(L.BatchNormalization(n_units))
        elif normalize == 'LN':
            net.append(L.LayerNormalization())
        net.append(mF.get_function(activation))
        net.append(partial(F.dropout, ratio=dropout))
    net.append(L.Linear(n_out))
    return net


class MovingAverageSubtractor(L.BatchNormalization):
    def __init__(self, size=None, decay=0.9, eps=2e-05, dtype=None, axis=None, initial_avg_mean=None):
        super().__init__(size=size, decay=decay, eps=eps, dtype=dtype, use_gamma=False, use_beta=False,
                         initial_gamma=None, initial_beta=None, axis=axis, initial_avg_mean=initial_avg_mean, initial_avg_var=None)

    def _initialize_params(self, shape):
        super()._initialize_params(shape)
        delattr(self, "avg_var")
        delattr(self, "N")

    def forward(self, x):

        with cuda.get_device_from_id(self._device_id):
            gamma = avg_var = chainer.as_variable(self.xp.ones(
                self.avg_mean.shape, dtype=x.dtype))
            beta = chainer.as_variable(self.xp.zeros(
                self.avg_mean.shape, dtype=x.dtype))

        if configuration.config.train:
            decay = self.decay
            F.batch_normalization(
                x, gamma, beta, eps=self.eps, running_mean=self.avg_mean,
                running_var=avg_var, decay=decay)

        mean = chainer.as_variable(self.avg_mean)
        var = chainer.as_variable(avg_var)
        ret = F.fixed_batch_normalization(
            x, gamma, beta, mean, var, self.eps)

        return ret


class NormalizedEmbedID(L.EmbedID):
    def forward(self, x):
        return F.normalize(super().forward(x))


class EmbDecID(L.EmbedID):
    def __init__(self, in_size, out_size, initialW=None,
                 ignore_label=None, dec_mask_idx=None):
        super().__init__(in_size, out_size, initialW, ignore_label)

        if dec_mask_idx is not None:
            dec_mask = np.zeros(in_size, dtype=self.W.dtype)
            dec_mask[dec_mask_idx] = -np.inf
            self.add_persistent("dec_mask", dec_mask)

    def forward(self, x, decode=False):
        if not decode:
            return super().forward(x)
        else:
            out = F.linear(x, self.W)
            if hasattr(self, "dec_mask"):
                out += self.dec_mask
            return out


class MovingAverageNormalization(L.BatchNormalization):

    def __init__(self, size=None, decay=0.9, eps=2e-5, dtype=None,
                 initial_gamma=None, initial_beta=None, axis=None,
                 initial_avg_mean=None, initial_avg_var=None):

        super().__init__(
            size=size, decay=decay, eps=eps,
            dtype=dtype, use_gamma=False, use_beta=False,
            initial_gamma=initial_gamma, initial_beta=initial_beta, axis=axis,
            initial_avg_mean=initial_avg_mean, initial_avg_var=initial_avg_var)

    def forward(self, x):
        ret = super().forward(x)
        if chainer.config.train:
            with chainer.using_config("train", False):
                ret = super().forward(x)

        return ret

    def inverse_transform(self, x):

        if not hasattr(self, "avg_mean"):
            return x
        elif self.avg_mean is None:
            return x

        reshape_axis = []
        counter = 0
        for i in range(x.ndim):
            if i in self.axis:
                reshape_axis.append(1)
            else:
                reshape_axis.append(self.avg_var.shape[counter])
                counter += 1

        avg_mean = self.avg_mean.reshape(reshape_axis)
        avg_var = self.avg_var.reshape(reshape_axis)
        ret = x * (avg_var ** 0.5) + avg_mean

        return ret

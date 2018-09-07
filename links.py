
from functools import partial
from math import ceil

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from . import functions as mF


class Convolution1D(L.Convolution2D):

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1):
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None
        super().__init__(in_channels, out_channels, (ksize, 1), (stride, 1),
                         (pad, 0), nobias, initialW, initial_bias, dilate=(dilate, 1), groups=groups)

    def __call__(self, x):
        """
        x is 3D ndarray（batch size  x input chanel x length)
        """
        x = F.reshape(x, (x.shape[0], x.shape[1], -1, 1))
        x = super().__call__(x)
        x = F.reshape(x, (x.shape[0], x.shape[1], -1))
        return x


class TSRegressor(L.Classifier):

    def __init__(self, predictor, lossfun=mF.sum_absolute_error):
        super().__init__(predictor, lossfun=lossfun)
        self.compute_accuracy = False

    def __call__(self, *args):
        """
        Args:
            args (list of ~chainer.Variable): Input minibatch.

        Returns:
            ~chainer.Variable: Loss value.
        """

        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        batch = len(t)
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(F.concat(self.y, axis=0), F.concat(t, axis=0)) / batch
        chainer.reporter.report({'loss': self.loss}, self)
        return self.loss


class Regressor(L.Classifier):

    def __init__(self, predictor, lossfun=F.mean_squared_error):
        super(Regressor, self).__init__(predictor, lossfun=lossfun)
        self.compute_accuracy = False


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
            self.conv = Convolution1D(
                in_channels, out_channels, ksize, stride, pad, True, initialW, initial_bias)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        return self.bn(self.conv(x), finetune=self.finetune)


class DilatedConvolution1D(L.DilatedConvolution2D):

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, dilate=1, nobias=False, initialW=None, initial_bias=None):

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        super().__init__(in_channels, out_channels, (ksize, 1),
                         (stride, 1), (pad, 0), dilate, nobias, initialW, initial_bias)

    def __call__(self, x):
        # x is 3D ndarray（batch size  x input chanel x length)
        x = F.reshape(x, (x.shape[0], x.shape[1], -1, 1))
        x = super().__call__(x)
        x = F.reshape(x, (x.shape[0], x.shape[1], -1))
        return x


class Deconvolution1D(L.DeconvolutionND):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, nobias=False, outsize=None, initialW=None, initial_bias=None):
        super().__init__(1, in_channels, out_channels, ksize,
                         stride, pad, nobias, outsize, initialW, initial_bias)


class Convolution3D(L.ConvolutionND):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, cover_all=False):
        super().__init__(3, in_channels, out_channels, ksize,
                         stride, pad, nobias, initialW, initial_bias, cover_all)


class HighWayConv1D(Convolution1D):
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


class GLU1D(chainer.Chain):
    """
    Deep Voice 3's Gated Linear Unit
    """
    def __init__(self, in_out_channels, ksize, dilate=1, dropout=0.0, use_cond=True, causal=True, residual=True):
        self.causal = causal
        self.pad = (ksize - 1) * dilate
        if not self.causal:
            self.pad //= 2

        self.dilate = dilate
        self.ksize = ksize
        self.dropout = dropout
        self.residual = residual
        super().__init__()
        with self.init_scope():
            self.conv = DilatedConvolution1D(
                in_out_channels * 2, ksize, dilate=dilate, pad=self.pad)
            if use_cond:
                self.conv_cond = Convolution1D(in_out_channels, 1)

    def __call__(self, x, cond=None):
        length = x.shape[-1]
        h = self.conv(F.dropout(x, self.dropout))[..., :length]
        h, hg = F.split_axis(h, 2, axis=1)
        if hasattr(self, 'conv_cond') and cond is not None:
            h += mF.softsign(self.conv_cond(cond)[..., :length])

        out = h * F.sigmoid(hg)
        if self.residual:
            out = (out + x) * np.sqrt(0.5)
        return out

    def init_que(self):
        if not self.causal:
            raise Exception
        self.que = []

    def fast_forward(self, x, cond=None):
        self.que.insert(0, x)
        x_tmp = F.concat(self.que[::-1], -1)
        length = x.shape[-1]
        h = self.conv(x_tmp)[..., :-self.pad][..., -1:]
        self.que = self.que[:self.pad]
        h, hg = F.split_axis(h, 2, axis=1)
        if hasattr(self, 'conv_cond') and cond is not None:
            h += mF.softsign(self.conv_cond(cond)[..., :length])

        out = h * F.sigmoid(hg)
        if self.residual:
            out = (out + x) * np.sqrt(0.5)
        return out

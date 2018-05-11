
from math import ceil

import chainer
import chainer.functions as F
import chainer.links as L

from . import functions as mF


class Convolution1D(L.Convolution2D):

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1):
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None
        super().__init__(in_channels, out_channels, (ksize, 1), (stride, 1),
                         (pad, 0), nobias, initialW, initial_bias, dilate=dilate, groups=groups)

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
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(F.concat(self.y, axis=0), F.concat(t, axis=0))
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


class HighWayConv1D(DilatedConvolution1D):
    def __init__(self, inout_channels, ksize, dilate=1, nobias=False, initialW=None, initial_bias=None, causal=False):
        if causal:
            pad = (ksize - 1) * dilate
        else:
            pad = ceil((ksize - 1) * dilate / 2)
        super().__init__(inout_channels, inout_channels * 2, ksize=ksize, stride=1, pad=pad,
                         dilate=dilate, nobias=nobias, initialW=initialW, initial_bias=initial_bias)

    def __call__(self, x):
        length = x.shape[-1]
        h1 = super().__call__(x)[..., :length]
        h2, h3 = F.split_axis(h1, 2, 1)
        h4 = F.sigmoid(h2)
        return h4 * h3 + (1 - h4) * x


class Deconvolution1D(L.DeconvolutionND):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, nobias=False, outsize=None, initialW=None, initial_bias=None):
        super().__init__(1, in_channels, out_channels, ksize, stride, pad, nobias, outsize, initialW, initial_bias)

class Convolution3D(L.ConvolutionND):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, cover_all=False):
        super().__init__(3, in_channels, out_channels, ksize, stride, pad, nobias, initialW, initial_bias, cover_all)
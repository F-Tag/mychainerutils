
import chainer
import chainer.functions as F
import numpy as np
from librosa.util import frame


def swish(x):
    return x * F.sigmoid(x)


def do_nothing(x):
    return x


def get_function(function):

    if function == 'swish':
        return swish

    elif function == 'softsign':
        return softsign

    elif function == 'do_nothing':
        return do_nothing

    elif isinstance(function, str):
        return getattr(F, function)

    else:
        return function


def pad_sequence_1d(xs, length=None, padding=0):
    return F.swapaxes(F.pad_sequence(xs), 1, 2)


def arr2list(arr, length=None):
    xs = F.separate(F.swapaxes(arr, 1, 2))

    if length is not None:
        assert len(xs) == len(length)
        xs = [x[:l] for x, l in zip(xs, length)]
    return xs


def sum_absolute_error(x0, x1):
    return F.sum(F.absolute_error(x0, x1))


def sum_squared_error(x0, x1):
    return F.sum(F.squared_difference(x0, x1))


def add_noise(h, sigma=0.2):
    # https://github.com/chainer/chainer/blob/master/examples/dcgan/net.py
    h = chainer.as_variable(h)
    xp = h.xp
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h


def stft(x, frame_length=1024, hop_length=512):
    # ..., FFT axis
    if not isinstance(x, chainer.Variable):
        x = chainer.as_variable(x)
    xp = x.xp
    pad_len = (x.shape[-1] // hop_length - frame_length //
               hop_length + 1) * hop_length + frame_length
    pad = pad_len - x.shape[-1]
    if pad > 0:
        shape = list(x.shape)
        pad = xp.zeros(shape[:-1] + [pad]).astype(x.dtype)
        x = F.concat((x, pad), -1)
    index = frame(np.arange(x.shape[-1]), frame_length, hop_length).T
    tmp = x[..., index] * xp.hamming(frame_length).astype(x.dtype)
    yr, yi = F.fft((tmp, xp.zeros(tmp.shape).astype(x.dtype)))
    return yr[..., :frame_length // 2 + 1], yi[..., :frame_length // 2 + 1]


def power_loss(x, t, frame_length=1024, hop_length=512, time_axis_mean=False):
    # ..., FFT axis
    Xr, Xi = stft(x, frame_length, hop_length)
    Xa = Xr ** 2 + Xi ** 2
    Tr, Ti = stft(t, frame_length, hop_length)
    Ta = Tr ** 2 + Ti ** 2

    if time_axis_mean:
        Xa = F.average(Xa, -1)
        Ta = F.average(Ta, -1)

    return F.mean_squared_error(Xa, Ta)


def softsign(x):
    return x / (1.0 + F.absolute(x))


def bilinear_interpolation_1d(x, rate=None, shape=None):
    length = x.shape[-1]
    if rate is not None:
        shape = int(length * rate)
    else:
        if shape is None:
            raise Exception('rate or shape')
        else:
            shape = int(shape)

    x = x[..., None]
    x = F.resize_images(x, (shape, 1))
    return F.squeeze(x, -1)


def gated_activation(x, activation=F.tanh):
    arr1, arr2 = F.split_axis(x, 2, axis=1)
    return F.sigmoid(
        arr1) * (activation(
            arr2) if activation is not None else arr2)


def delta_feature(x, order=4, static=True, delta=True, deltadelta=True):

    length = None
    dim2_flag = False
    if isinstance(x, list) or isinstance(x, tuple):
        length = [len(arr) for arr in x]
        x = F.pad_sequence(x).transpose(0, 2, 1)

    elif x.ndim == 2:
        x = x[None].transpose(0, 2, 1)
        dim2_flag = True

    x = F.expand_dims(x, 1)
    xp = x.xp
    dtype = x.dtype

    ws = []
    if order == 2:
        if static:
            ws.append(np.array((0, 1, 0)))
        if delta:
            ws.append(np.array((-1, 0, 1)) / 2)
        if deltadelta:
            ws.append(np.array((1.0, -2.0, 1.0)))
        pad = 1

    elif order == 4:
        if static:
            ws.append(np.array((0, 0, 1, 0, 0)))
        if delta:
            ws.append(np.array((1, -8, 0, 8, -1)) / 12)
        if deltadelta:
            ws.append(np.array((-1, 16, -30, 16, -1)) / 12)
        pad = 2

    else:
        raise ValueError(f"order: {order}")
    W = xp.array(np.expand_dims(np.vstack(ws), (1, 2))).astype(dtype)

    pad_width = [(0, 0)]*3 + [(pad, pad)]
    x = F.pad(x, pad_width, mode="reflect")
    out = F.convolution_2d(x, W)
    B, T = out.shape[0], out.shape[-1]
    out = out.reshape(B, -1, T)

    if length is not None:
        out = [arr[:l] for arr, l in zip(F.separate(
            out.transpose(0, 2, 1), axis=0), length)]

    elif dim2_flag:
        out = F.squeeze(out.transpose(0, 2, 1), 0)

    return out


def to_finite(arr):
    arr = chainer.as_variable(arr)
    xp = arr.xp
    dtype = arr.dtype

    condition = xp.isfinite(arr.array)
    if condition.all():
        ret = arr
    else:
        ret = F.where(condition,
                      arr,
                      xp.array(0.0, dtype=dtype))
    return ret


class GradientScale(chainer.function_node.FunctionNode):

    def __init__(self, lmbd):
        self.lmbd = lmbd

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs
        return self.lmbd * gy,


def gradient_scale(x, lmbd=-1.0):
    """
    Scale gradient in bakprop.
    if lmbd = -1.0, this is equivalent to gradient reversal layer.
    """
    y, = GradientScale(lmbd).apply((x,))
    return y


def irfft(real, imag):
    real = F.concat((real, F.flip(real[..., 1:-1], -1)), -1)
    imag = F.concat((imag, F.flip(-imag[..., 1:-1], -1)), -1)
    ret, _ = F.ifft((real, imag))
    return ret


def depthwise_normalization(x):
    x = chainer.as_variable(x)
    dim = x.shape[1]
    xp = x.xp
    dtype = x.dtype

    gamma = xp.ones(dim, dtype=dtype)
    beta = xp.zeros(dim, dtype=dtype)
    return F.group_normalization(x, dim, gamma, beta)


def stats_pooling(x, mean_only=False):
    mean = F.mean(x, axis=-1)
    if mean_only:
        return mean
    var = F.mean(x**2, axis=-1) - mean**2
    std = F.sqrt(F.clip(var, 0, np.inf))
    return F.concat((mean, std), axis=1)

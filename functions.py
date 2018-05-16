
import chainer
import chainer.functions as F
import numpy as np
from librosa.util import frame


def swish(x):
    return x * F.sigmoid(x)


def do_nothing(x):
    return x


def get_function(name):

    if name == 'swish':
        return swish

    elif name == 'do_nothing':
        return do_nothing

    else:
        return getattr(F, name)


def pad_sequence_1d(xs, length=None, padding=0):
    return F.swapaxes(F.pad_sequence(xs), 1, 2)


def arr2list(arr, length):
    xs = F.separate(F.swapaxes(arr, 1, 2))
    assert len(xs) == len(length)
    return [x[:l] for x, l in zip(xs, length)]


def sum_absolute_error(x0, x1):
    return F.sum(F.absolute_error(x0, x1))


def sum_squared_error(x0, x1):
    return F.sum(F.squared_difference(x0, x1))


def l1_bd(y, t):
    return sum_absolute_error(y, t) + F.bernoulli_nll(t, y)


def add_noise(h, sigma=0.2):
    # https://github.com/chainer/chainer/blob/master/examples/dcgan/net.py
    xp = chainer.cuda.get_array_module(h.array)
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

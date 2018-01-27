
import chainer.functions as F


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
    xs = F.separate(arr)
    assert len(xs) == len(length)
    return [F.transpose(x)[:l] for x, l in zip(xs, length)]

def sum_absolute_error(x0, x1):
    return F.sum(F.absolute_error(x0, x1))

def sum_squared_error(x0, x1):
    return F.sum(F.squared_difference(x0, x1))
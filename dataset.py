
import chainer

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
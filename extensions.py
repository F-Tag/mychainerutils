from __future__ import division

from chainer.training import extension


class TransformerShift(extension.Extension):

    """
    Learning rate scheduling used in "attention is all you need" 
    (https://arxiv.org/abs/1706.03762)

    Args:
        attr (str): Name of the optimizer attribute to adjust.
        d_model (int): channels or units of model
            of one training.
        warmup_steps (int): the number of the iterations while warmup.
        optimizer (~chainer.Optimizer): Target optimizer object.
            If it is None, the main optimizer of the trainer is used.

    """

    def __init__(self, attr, d_model, warmup_steps=4000, optimizer=None):
        self._attr = attr
        self._warmup_steps = warmup_steps
        self._optimizer = optimizer
        self._t = 0
        self._base = d_model ** (-0.5)

    def initialize(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        setattr(optimizer, self._attr, 0.0)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._optimizer or \
            trainer.updater.get_optimizer('main')

        # eq. (3)
        value = self._base * min(self._t ** (-0.5),
                                 self._t*self._warmup_steps ** (-1.5))
        setattr(optimizer, self._attr, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)


class NoamScheme(extension.Extension):

    """
    Tensor2Tensor's Noam scheme

    Args:
        attr (str): Name of the optimizer attribute to adjust.
        init_lr (float): channels or units of model
            of one training.
        c (float): 
        warmup_steps (int): the number of the iterations while warmup.
        optimizer (~chainer.Optimizer): Target optimizer object.
            If it is None, the main optimizer of the trainer is used.

    """

    def __init__(self, attr, init_lr=0.002, c=0.5, warmup_steps=4000, optimizer=None):
        self._attr = attr
        self._warmup_steps = warmup_steps
        self._optimizer = optimizer
        self._t = 0
        self._init_lr = init_lr
        self._c = c

    def initialize(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        setattr(optimizer, self._attr, 0.0)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._optimizer or \
            trainer.updater.get_optimizer('main')

        value = self._init_lr * self._warmup_steps**self._c * \
            min((self._t) * self._warmup_steps**-
                (1.0+self._c), self._t**-self._c)
        setattr(optimizer, self._attr, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)

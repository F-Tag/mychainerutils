"""
modified:
https://gist.github.com/hrsma2i/9c6514e94cd5e802d9e216aef2bcfe59
"""

from math import cos, pi

import numpy

from chainer.training import extension

class CosineAnnealing(extension.Extension):
    def __init__(self, attr, vmax, vmin=0, T_0=1, T_mult=1,
                 optimizer=None):
        super(CosineAnnealing, self).__init__()
        self._attr = attr
        self.vmax = vmax
        self.vmin = vmin
        self.T_0 = T_0
        self.T_mult = T_mult
        self.optimizer = optimizer
        self._t = 0
        self._last_lr = None
        self._last_t_i = None
        self._last_T_i = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)

        if self._last_lr is not None: # resuming from a snapshot
            self._update_value(optimizer, self._last_lr)
            self._update_t_i(optimizer, self._last_t_i)
            self._update_T_i(optimizer, self._last_T_i)
        else:
            self._update_value(optimizer, self.vmax)
            self._update_t_i(optimizer, 1)
            self._update_T_i(optimizer, self.T_0)

    def __call__(self, trainer):
        self._t += 1

        _t     = self._t # global iteration
        vmax = self.vmax
        vmin = self.vmin
        T_0    = self.T_0
        T_mult = self.T_mult

        optimizer = self._get_optimizer(trainer)
        t_cmsm = _t - numpy.cumsum([T_0*(T_mult**i) for i in range(10)])
        # cycle
        i = numpy.where(t_cmsm<0)[0][0]
        # local iteration (in cycle i, at global iteration t)
        T_i = T_0 * (T_mult**i)
        t_i = int(_t - (T_i - T_0)/max(T_mult - 1, 1)) + 1
        lr = vmin + (vmax-vmin) * 0.5 * (1 + cos(pi*(t_i-1)/T_i)) 

        self._update_value(optimizer, lr)
        self._update_t_i(optimizer, t_i)
        self._update_T_i(optimizer, T_i)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_lr = serializer('_last_lr', self._last_lr)
        self._last_t_i = serializer('_last_t_i', self._last_t_i)
        self._last_T_i = serializer('_last_T_i', self._last_T_i)
        if isinstance(self._last_lr, numpy.ndarray):
            self._last_lr = numpy.asscalar(self._last_lr)

    def _get_optimizer(self, trainer):
        return self.optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_lr = value

    def _update_t_i(self, optimizer, value):
        setattr(optimizer, 't_i', value)
        self._last_t_i = value

    def _update_T_i(self, optimizer, value):
        setattr(optimizer, 'T_i', value)
        self._last_T_i = value
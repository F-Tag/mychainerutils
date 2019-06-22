import chainer
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer import link_hook
from chainer import variable
from warnings import warn


def _norm(V, axis=1, eps=1e-5):
    with chainer.using_config('enable_backprop', False):
        return F.sqrt(F.sum(V * V, axis, True)).array + eps


class WeightNormalization(link_hook.LinkHook):
    r"""Weight Normalization link hook implementation.
    """

    name = 'WeightNormalization'

    def __init__(self, axis=1, weight_name='W', eps=1e-5, name=None):
        self.eps = eps
        self.weight_name = weight_name
        self.g_name = weight_name + '_g'
        self.v_name = weight_name + '_v'
        self._initialied = False
        self.axis = axis

        if name is not None:
            self.name = name

    def added(self, link):
        if isinstance(
            link, (
                L.Deconvolution1D, L.Deconvolution2D,
                L.Deconvolution3D, L.DeconvolutionND)):
            if self.axis == 1:
                warn("Please pay attention to the axis when "
                    "weight normalization is applied to Deconvolution.")

        with link.init_scope():
            setattr(link, self.g_name, variable.Parameter())
            setattr(link, self.v_name, variable.Parameter())

    def deleted(self, link):
        self._set_weight(link)
        W = getattr(link, self.weight_name).array
        delattr(link, self.g_name)
        delattr(link, self.v_name)
        delattr(link, self.weight_name)

        with link.init_scope():
            setattr(link, self.weight_name, variable.Parameter(W))

    def forward_preprocess(self, cb_args):
        if configuration.config.train:
            link = cb_args.link
            input_variable = cb_args.args[0]
            if not self._initialied:
                self._prepare_parameters(link, input_variable)

            self._set_weight(link)

    def forward_postprocess(self, cb_args):
        if configuration.config.train:
            link = cb_args.link
            weight = getattr(link, self.weight_name).array
            with link.init_scope():
                setattr(link, self.weight_name, variable.Parameter(weight))

    def _prepare_parameters(self, link, input_variable=None):
        if getattr(link, self.weight_name).array is None:
            if input_variable is not None:
                link._initialize_params(input_variable.shape[1])
        initialW = getattr(link, self.weight_name)
        g = _norm(initialW, self.axis)
        V = initialW / g

        getattr(link, self.g_name).array = g
        getattr(link, self.v_name).array = V.array

        self._initialied = True

    def _set_weight(self, link):
        V = getattr(link, self.v_name)
        g = getattr(link, self.g_name)
        weight = V / _norm(V, self.axis) * g
        delattr(link, self.weight_name)
        setattr(link, self.weight_name, weight)

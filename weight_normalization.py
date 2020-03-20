
import chainer.links as L
from chainer import link_hook, variable


def get_axes_except_myself(axis, ndim):
    if axis is None:
        return None
    axes = list(range(ndim))
    axes.pop(axis)
    return tuple(axes)


def get_norm(xp, v, eps, axis, keepdims=False):
    """get L2 norm.

    Args:
        xp (numpy or cupy):
        v (numpy.ndarray or cupy.ndarray or chainerx.ndarray)
        eps (float): Epsilon value for numerical stability.

    Returns:
        :class:`xp.ndarray`

    """
    axes = get_axes_except_myself(axis, v.ndim)
    norm = xp.sqrt(xp.sum(v * v, axis=axes, keepdims=keepdims))
    return norm + eps


class WeightNormalization(link_hook.LinkHook):
    name = 'WeightNormalization'

    def __init__(self, eps=1e-6, weight_name='W', name=None):
        self.eps = eps
        self.weight_name = weight_name
        self.g_name = weight_name + '_g'
        self._initialized = False
        self.axis = 0
        if name is not None:
            self.name = name

    def __enter__(self):
        raise NotImplementedError(
            'This hook is not supposed to be used as context manager.')

    def __exit__(self):
        raise NotImplementedError

    def added(self, link):
        # Define axis and register ``u`` if the weight is initialized.
        if not hasattr(link, self.weight_name):
            raise ValueError(
                'Weight \'{}\' does not exist!'.format(self.weight_name))
        if isinstance(link, (L.Deconvolution2D, L.DeconvolutionND)):
            self.axis = 1

        with link.init_scope():
            setattr(link, self.g_name, variable.Parameter())

        if getattr(link, self.weight_name).array is not None:
            self._prepare_parameters(link)

    def deleted(self, link):
        # Remove g and set weight.
        normalized_weight = self.normalize_weight(link)
        getattr(link, self.weight_name).array = normalized_weight.array
        delattr(link, self.g_name)

    def forward_preprocess(self, cb_args):
        # This method normalizes target link's weight
        link = cb_args.link
        input_variable = cb_args.args[0]
        if not self._initialized:
            self._prepare_parameters(link, input_variable)
        weight = getattr(link, self.weight_name)
        # For link.W or equivalents to be chainer.Parameter
        # consistently to users, this hook maintains a reference to
        # the unnormalized weight.
        self.original_weight = weight
        # note: `normalized_weight` is ~chainer.Variable
        normalized_weight = self.normalize_weight(link)
        setattr(link, self.weight_name, normalized_weight)

    def forward_postprocess(self, cb_args):
        # Here, the computational graph is already created,
        # we can reset link.W or equivalents to be Parameter.
        link = cb_args.link
        setattr(link, self.weight_name, self.original_weight)

    def _prepare_parameters(self, link, input_variable=None):
        """Prepare one buffer and one parameter.

        Args:
            link (:class:`~chainer.Link`): Link to normalize spectrally.
            input_variable (:class:`~chainer.Variable`):
                The first minibatch to initialize weight.

        """
        if getattr(link, self.weight_name).array is None:
            if input_variable is not None:
                link._initialize_params(input_variable.shape[1])
        initialW = getattr(link, self.weight_name)
        if initialW.shape[self.axis] == 0:
            raise ValueError(
                'Expect {}.shape[{}] > 0'.format(self.weight_name, self.axis)
            )

        if getattr(link, self.g_name).array is None:
            g = get_norm(initialW.xp, initialW.array, self.eps, axis=self.axis)
            getattr(link, self.g_name).array = g

        self._initialized = True

    def normalize_weight(self, link):
        """Normalize target weight before every single forward computation."""
        weight_name, g_name = self.weight_name, self.g_name
        W = getattr(link, weight_name)
        g = getattr(link, g_name)

        # print(link.W.array.sum())
        norm = get_norm(W.xp, W.array, self.eps, axis=self.axis, keepdims=True)
        W = W / norm * g.reshape(norm.shape)

        return W

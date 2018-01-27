
import chainer
import chainer.functions as F
import chainer.links as L

from . import functions as mF


class Convolution1D(L.Convolution2D):

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None):

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        super().__init__(in_channels, out_channels, (ksize, 1), (stride, 1),
                         (pad, 0), nobias, initialW, initial_bias)

    def __call__(self, x):
        """
        x is 3D ndarray（batch size  x input chanel x length)
        """
        x = F.reshape(x, (x.shape[0], x.shape[1], -1, 1))
        x = super().__call__(x)
        x = F.reshape(x, (x.shape[0], x.shape[1], -1))
        return x


class ResBlock1D(chainer.Chain):
    """
    Residual Block 1D
    """

    def __init__(self, channels, ksize=3, stride=1, dropout=0.0, acfun='leaky_relu'):
        self.f = mF.get_function(acfun)
        self.dropout = dropout
        super().__init__()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(channels)
            self.bn2 = L.BatchNormalization(channels)
            self.conv1 = Convolution1D(
                channels, channels, ksize, stride, ksize // 2, nobias=True)
            self.conv2 = Convolution1D(
                channels, channels, ksize, stride, ksize // 2, nobias=True)

    def __call__(self, x, finetune=False):
        h = self.bn1(x, finetune=finetune)
        h = self.f(h)
        h = self.conv1(h)
        h = self.bn2(h, finetune=finetune)
        h = self.f(h)
        h = F.dropout(h, self.dropout)
        h = self.conv2(h)
        return h + x



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

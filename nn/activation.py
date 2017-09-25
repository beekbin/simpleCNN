
from __future__ import division
from __future__ import print_function
import numpy as np


class ActiveFunction(object):
    def __init__(self, name):
        self.name = name

    def forward(self, x, out=None):
        pass

    def backward(self, x, y, out=None):
        pass

    def get_name(self):
        return self.name

    def __str__(self):
        return self.name


class SigmoidActiveFunction(ActiveFunction):
    def __init__(self):
        super(self.__class__, self).__init__("sigmoid")
        return

    def forward(self, x, out=None):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, y, out=None):
        return y * (1-y)


class TanhActiveFunction(ActiveFunction):
    def __init__(self):
        super(self.__class__, self).__init__("tanh")
        return

    def forward(self, x, out=None):
        if out is None:
            return np.tanh(x)
        np.tanh(x, out=out)
        return out

    def backward(self, x, y, out=None):
        if out is None:
            return 1 - y*y

        out.fill(1.0)
        out -= y*y
        return out


class ReluActiveFunction(ActiveFunction):
    def __init__(self):
        super(self.__class__, self).__init__("relu")
        return

    def forward(self, x, out=None):
        if out is None:
            return np.maximum(x, 0)
        np.maximum(x, 0, out=out)
        return out

    def backward(self, x, y, out=None):
        """TODO:
            return 1 if x > 0;
                   0.5 if x == 0;
                   0 otherwise;
        """
        if out is None:
            return 1.0

        out.fill(1.0)
        return out


class LinearActiveFunction(ActiveFunction):
    def __init__(self):
        super(self.__class__, self).__init__("linear")
        return

    def forward(self, x, out=None):
        if out is None:
            out = np.copy(x)
        else:
            np.copyto(out, x)
        return out

    def backward(self, x, y, out=None):
        if out is None:
            return 1.0

        out.fill(1.0)
        return out


class LeakyReluActiveFunction(ActiveFunction):
    """" f(x) = x, if x > 0;
         f(x) = alpha * x, otherwise; (alpha usually is 0.01)
    """
    def __init__(self, alpha=0.01):
        super(self.__class__, self).__init__("leaky-relu")
        self.alpha = alpha
        return

    def forward(self, x, out=None):
        if out is None:
            out = np.ones(x.shape)
        else:
            out.fill(1.0)

        out[x <= 0] = self.alpha
        out *= x
        return out

    def backward(self, x, y, out=None):
        if out is None:
            out = np.ones(x.shape)
        else:
            out.fill(1.0)

        out[x <= 0] = self.alpha
        return out


class SoftmaxActiveFunction(ActiveFunction):
    def __init__(self):
        super(self.__class__, self).__init__("softmax")
        return

    def forward(self, x, out=None):
        if out is None:
            out = np.exp(x)
        else:
            np.exp(x, out=out)
        d = np.sum(out)
        out /= d
        return out

    def backward(self, x, y, out=None):
        """let softmax layer do the actually backward calculation."""
        return 0.0


tanhFunc = TanhActiveFunction()
sigmoidFunc = SigmoidActiveFunction()
reluFunc = ReluActiveFunction()
linearFunc = LinearActiveFunction()
leakyReluFunc = LeakyReluActiveFunction(0.01)
softmaxFunc = SoftmaxActiveFunction()

from __future__ import print_function, division
import numpy as np

from activation import SigmoidActiveFunction
from activation import TanhActiveFunction
from activation import ReluActiveFunction
from activation import LeakyReluActiveFunction
from activation import SoftmaxActiveFunction


def vector_str(v):
    if not isinstance(v, np.ndarray):
        return "%.3f" % (v)

    result = "["
    for e in v:
        result += ("%.3f  "%(e))

    result += "]"
    return result


def test_sigmoid():
    afunc = SigmoidActiveFunction()
    print(afunc)
    x = np.array([0, 1, -1, 2, -2, 4, -4])
    y = afunc.forward(x)
    dy = afunc.backward(x, y)
    print("x: %s" % vector_str(x))
    print("y: %s" % vector_str(y))
    print("dy: %s" % vector_str(dy))
    afunc.backward(x, y, out=dy)
    print("dy: %s" % vector_str(dy))
    return


def test_tanh():
    afunc = TanhActiveFunction()
    print(afunc)
    x = np.array([0, 1, -1, 2, -2, 4, -4])
    y = afunc.forward(x)
    dy = afunc.backward(x, y)
    print("x: %s" % vector_str(x))
    print("y: %s" % vector_str(y))
    print("dy: %s" % vector_str(dy))
    afunc.backward(x, y, out=dy)
    print("dy: %s" % vector_str(dy))
    return


def test_relu():
    afunc = ReluActiveFunction()
    print(afunc)
    x = np.array([0, 1, -1, 2, -2, 4, -4])
    y = afunc.forward(x)
    dy = afunc.backward(x, y)
    print("x: %s" % vector_str(x))
    print("y: %s" % vector_str(y))
    print("dy: %s" % vector_str(dy))
    dy = np.zeros(x.shape)
    afunc.backward(x, y, out=dy)
    print("dy: %s" % vector_str(dy))
    return


def test_leaky_relu():
    afunc = LeakyReluActiveFunction(0.005)
    print(afunc)
    x = np.array([0, 1, -1, 2, -2, 4, -4])
    y = afunc.forward(x)
    dy = afunc.backward(x, y)
    print("x: %s" % vector_str(x))
    print("y: %s" % vector_str(y))
    print("dy: %s" % vector_str(dy))
    dy = np.zeros(x.shape)
    afunc.backward(x, y, out=dy)
    print("dy: %s" % vector_str(dy))
    return


def test_softmax():
    afunc = SoftmaxActiveFunction()
    print(afunc)
    x = np.array([0, 1, -1, 2, -2, 4, -4])
    y = afunc.forward(x)
    dy = afunc.backward(x, y)
    print("x: %s" % vector_str(x))
    print("y: %s" % vector_str(y))
    print("dy: %s" % vector_str(dy))
    dy = np.zeros(x.shape)
    afunc.backward(x, y, out=dy)
    print("dy: %s" % vector_str(dy))
    return


def main():
    test_sigmoid()
    test_tanh()
    test_relu()
    test_leaky_relu()
    test_softmax()
    return


if __name__ == "__main__":
    main()

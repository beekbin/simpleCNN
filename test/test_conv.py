from __future__ import division
from __future__ import print_function
import sys
from datetime import datetime
import numpy as np

sys.path.insert(0, "../")
from nn.conv_layer import ConvLayer
from nn.conv_layer import Kernel
from nn.nn_layer import InputLayer
from nn.nn_layer import HiddenLayer
from nn import activation


def get_kernels():
    result = []
    uuid = 1

    # 1. 3x3 kernels
    for i in range(8):
        func = activation.reluFunc
        if i % 3 == 0:
            func = activation.tanhFunc
        kernel = Kernel(3, func, uuid)
        result.append(kernel)
        uuid += 1

    # 2. 5x5 kernels
    for i in range(8):
        func = activation.reluFunc
        if i % 3 == 0:
            func = activation.tanhFunc
        kernel = Kernel(5, func, uuid)
        result.append(kernel)
        uuid += 1
    return result


def test_normal_active():
    img_input = InputLayer("fake_input", 784)
    h1 = HiddenLayer("h1", 784, activation.tanhFunc)
    h1.set_input_layer(img_input)
    h1.init()

    input_data = np.random.uniform(-1, 1, 784)
    img_input.feed(input_data)

    print("[%s] begin to active normal hidden layer." % (str(datetime.now())))
    for i in range(1000):
        h1.active()
    print("[%s] end of activation normal hidden layer." % (str(datetime.now())))
    return


def test_active():
    img_input = InputLayer("fake_input", 784)
    conv = ConvLayer("conv1")
    conv.set_kernels(get_kernels())
    conv.set_input_layer(img_input)

    input_data = np.random.uniform(-1, 1, 784)
    input_data = input_data.reshape((1, 28, 28))
    img_input.feed(input_data)
    conv.init()

    print("[%s] begin to active." % (str(datetime.now())))
    for i in range(1000):
        conv.active()
    print("[%s] end of activation." % (str(datetime.now())))
    return


def main():
    test_normal_active()
    test_active()
    return


if __name__ == "__main__":
    sys.exit(main())
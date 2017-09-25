from __future__ import division
from __future__ import print_function
import sys
import numpy as np
sys.path.insert(0, "../")
from nn.conv_layer import MaxPoolingLayer
from nn.nn_layer import InputLayer


def test_max_active():
    maxlayer = MaxPoolingLayer("maxpool1", 2, 2)

    inputlayer = InputLayer("input", 0)
    data = np.array(range(1, 50)).reshape((-1, 7, 7))
    data[0, 1, 2] = 1000
    inputlayer.feed(data)

    fakeOutput = InputLayer("output", 0)

    maxlayer.set_input_layer(inputlayer)
    maxlayer.set_next_layer(fakeOutput)

    maxlayer.init()
    maxlayer.active()
    print(maxlayer)
    print("\ninput data:")
    print(data)
    print("\nmax pooling result:")
    print(maxlayer.get_output())
    print("\nindex-keeper:")
    print(maxlayer.input_keeper)

    return


def test_max_active2():
    maxlayer = MaxPoolingLayer("maxpool1", 2, 2)

    inputlayer = InputLayer("input", 0)
    data = np.array(range(1, 37)).reshape((-1, 6, 6))
    data[0, 1, 2] = 1000
    inputlayer.feed(data)

    fakeOutput = InputLayer("output", 0)

    maxlayer.set_input_layer(inputlayer)
    maxlayer.set_next_layer(fakeOutput)

    maxlayer.init()
    maxlayer.active()
    print(maxlayer)
    print("\ninput data:")
    print(data)
    print("\nmax pooling result:")
    print(maxlayer.get_output())
    print("\norigin pooling result:")
    print(maxlayer.output)
    print("\nindex-keeper:")
    print(maxlayer.input_keeper)

    return


def test_max_delta():
    maxlayer = MaxPoolingLayer("maxpool1", 2, 2)

    inputlayer = InputLayer("input", 0)
    data = np.array(range(1, 17)).reshape((-1, 4, 4))
    data[0, 1, 2] = 1000
    inputlayer.feed(data)
    input_delta = np.zeros(data.shape)
    delta = np.array(range(1, 5)).reshape((-1, 2, 2))

    fakeOutput = InputLayer("output", 0)

    maxlayer.set_input_layer(inputlayer)
    maxlayer.set_next_layer(fakeOutput)

    maxlayer.init()
    maxlayer.active()
    maxlayer.delta = delta
    maxlayer.calc_input_delta(input_delta)

    print(maxlayer)
    print("\ninput data:")
    print(data)
    print("\nmax pooling result:")
    print(maxlayer.get_output())
    print("\norigin pooling result:")
    print(maxlayer.output)
    print("\nindex-keeper:")
    print(maxlayer.input_keeper)
    print("\ninput_delta")
    print(input_delta)

    return


def test_max_delta2():
    maxlayer = MaxPoolingLayer("maxpool1", 2, 2)

    inputlayer = InputLayer("input", 0)
    data = np.array(range(1, 26)).reshape((-1, 5, 5))
    data[0, 1, 2] = 1000
    inputlayer.feed(data)
    input_delta = np.zeros(data.shape)
    delta = np.array(range(1, 10)).reshape((-1, 3, 3))

    fakeOutput = InputLayer("output", 0)

    maxlayer.set_input_layer(inputlayer)
    maxlayer.set_next_layer(fakeOutput)

    maxlayer.init()
    maxlayer.active()
    maxlayer.delta = delta
    maxlayer.calc_input_delta(input_delta)

    print(maxlayer)
    print("\ninput data:")
    print(data)
    print("\nmax pooling result:")
    print(maxlayer.get_output())
    print("\norigin pooling result:")
    print(maxlayer.output)
    print("\nindex-keeper:")
    print(maxlayer.input_keeper)
    print("\ninput_delta")
    print(input_delta)

    return


def test_max_delta3():
    maxlayer = MaxPoolingLayer("maxpool1", 2, 2)

    inputlayer = InputLayer("input", 0)
    data = np.array(range(1, 33)).reshape((-1, 4, 4))
    data[0, 1, 2] = 1000
    data[1, 2, 1] = 1000
    inputlayer.feed(data)
    input_delta = np.zeros(data.shape)
    delta = np.array(range(1, 9)).reshape((-1, 2, 2))

    fakeOutput = InputLayer("output", 0)

    maxlayer.set_input_layer(inputlayer)
    maxlayer.set_next_layer(fakeOutput)

    maxlayer.init()
    maxlayer.active()
    maxlayer.delta = delta
    maxlayer.calc_input_delta(input_delta)

    print(maxlayer)
    print("\ninput data:")
    print(data)
    print("\nmax pooling result:")
    print(maxlayer.get_output())
    print("\norigin pooling result:")
    print(maxlayer.output)
    print("\nindex-keeper:")
    print(maxlayer.input_keeper)
    print("\ninput_delta")
    print(input_delta)

    return


def main():
    # test_max_active()
    # test_max_active2()
    # test_max_delta()
    # test_max_delta2()
    test_max_delta3()
    return

if __name__ == "__main__":
    sys.exit(main())

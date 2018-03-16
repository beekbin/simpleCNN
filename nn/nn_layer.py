from __future__ import print_function
from __future__ import division
import numpy as np
import math


def myrandom_array2d(d1, d2):
    return np.random.randn(d1, d2)


def myrandom_vector(d1):
    return np.random.randn(d1)


# init the weight with care
#  https://stats.stackexchange.com/questions/204114/deep-neural-network-weight-initialization?rq=1
#  https://arxiv.org/abs/1206.5533  Practical Recommendations for Gradient-Based Training of Deep Architectures
def sigmoid_init_weights(d1, d2):
    num = d1 + d2
    r = math.sqrt(6.0/num)
    tmp = np.random.uniform(-r, r, num)
    return tmp.reshape((d1, d2))


def tanh_init_weights(d1, d2):
    tmp = sigmoid_init_weights(d1, d2)
    tmp *= 4.0
    return tmp


def calc_softmax(z):
    tmp = np.exp(z)
    total = sum(tmp)
    return tmp/total


class Layer(object):
    def __init__(self, name, size):
        self.name = name
        self.size = size
        # activation function
        self.func = None
        # the output of current layer, usually is a vector of the activated result
        self.output = None
        self.input_layer = None
        self.next_layer = None
        self.lambda2 = 0
        return

    def init(self):
        """init the weight matrix"""
        pass

    def set_lambda2(self, l2):
        self.lambda2 = l2
        return

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer
        return

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer
        return

    def get_output(self):
        return self.output

    def get_size(self):
        return self.size

    def __str__(self):
        return "%d\t%s" % (self.size, self.name)

    def detail_info(self):
        fan_in = 0
        if self.input_layer is not None:
            fan_in = self.input_layer.get_size()

        if self.func is None:
            funcName = "None"
        else:
            funcName = self.func.get_name()
        msg = "[%d, %d],l2=%.5f, activation=[%s], %s" % (fan_in, self.size, self.lambda2, funcName, self.name)
        return msg


class InputLayer(Layer):
    def __init__(self, name, size):
        super(InputLayer, self).__init__(name, size)
        return

    def init(self):
        """do nothing"""
        # self.output = np.zeros(self.size)
        return

    def feed(self, data):
        self.output = data
        return


class ActiveLayer(Layer):
    def __init__(self, name, size):
        super(ActiveLayer, self).__init__(name, size)
        self.weights = None
        self.bias = None
        self.z = None
        self.delta = None
        self.delta_weights = None
        self.delta_bias = None
        return

    def init(self):
        fan_in = self.input_layer.get_size()
        self.weights = sigmoid_init_weights(fan_in, self.size)
        self.bias = myrandom_vector(self.size)

        # forward results
        self.z = np.zeros(self.size)
        self.output = np.zeros(self.size)

        # backward results
        self.delta = np.zeros(self.size)
        self.delta_weights = np.zeros((fan_in, self.size))
        self.delta_bias = np.zeros(self.size)
        return

    def clear_delta(self):
        self.delta.fill(0.0)
        self.delta_weights.fill(0.0)
        self.delta_bias.fill(0.0)
        return

    def active(self):
        pass

    def calc_error(self):
        pass

    def get_delta(self):
        return self.delta

    def get_weights(self):
        return self.weights

    def update_weights(self, lr):
        if self.lambda2 > 0:
            self.delta_weights += (self.lambda2 * self.weights)

        self.weights -= lr * self.delta_weights
        self.bias -= lr * self.delta_bias

        self.clear_delta()
        return

    def update_weights_batch(self, lr, batch_size):
        if batch_size > 1:
            batch_size = float(batch_size)
            self.delta_weights /= batch_size
            self.delta_bias /= batch_size

        self.update_weights(lr)
        return


class SoftmaxOutputLayer(ActiveLayer):
    def __init__(self, name, size):
        super(SoftmaxOutputLayer, self).__init__(name, size)
        return

    def init(self):
        self.output = np.zeros(self.size)
        return

    def active(self):
        x = self.input_layer.get_output()
        z = x - np.max(x)
        np.exp(z, out=self.output)
        d = np.sum(self.output)
        self.output /= d
        return

    def calc_error(self, labels):
        self.delta = self.output - labels
        return

    def calc_input_delta(self, delta):
        np.copyto(delta, self.delta)
        return

    def calc_cost(self, labels):
        i = np.argmax(labels)
        y = self.output[i]
        if y < 0.00000001:
            return 10000000
        elif y > 0.99999999:
            return 0

        return -1*math.log(y)

    def update_weights(self, lr):
        """do nothing"""
        return


class HiddenLayer(ActiveLayer):
    def __init__(self, name, size, activefunc):
        super(HiddenLayer, self).__init__(name, size)
        self.func = activefunc
        return

    def init(self):
        super(HiddenLayer, self).init()
        if self.func.get_name() == "tanh":
            self.weights *= 4

        return

    def active(self):
        # print("[220] [active] %s" % (self,))
        x = self.input_layer.get_output()
        np.dot(x, self.weights, out=self.z)
        self.z += self.bias

        self.func.forward(self.z, out=self.output)
        # if check_toobig(self.output):
        #    print("%s too big" % (self.name,))
        return

    def calc_input_delta(self, delta):
        np.dot(self.weights, self.delta, out=delta)
        return

    def calc_error(self):
        # 1. calc delta
        self.next_layer.calc_input_delta(self.delta)
        self.delta *= self.func.backward(self.z, self.output)

        # 2. calc delta_weights
        x = self.input_layer.get_output()
        self.delta_weights += np.dot(x.reshape((-1, 1)), self.delta.reshape((1, -1)))
        self.delta_bias += self.delta
        return


def check_toobig(y):
    for i in range(y.shape[0]):
        z = y[i]
        if z*z > 10000:
            print("z=%s" % (z, ))
            return True
    return False

from __future__ import division
from __future__ import print_function
import math
import numpy as np
from nn_layer import Layer


def init_nd_weights(shape, c=1.0):
    num = 1
    for i in shape:
        num *= i

    r = math.sqrt(6.0/num)
    tmp = np.random.uniform(-r, r, num)
    tmp *= c
    return tmp.reshape(shape)


def flip180X(kernel, result):
    """flip the kernel matrix 180"""
    shape = kernel.shape
    m = shape[0] - 1
    n = shape[1] - 1

    for i in range(shape[0]):
        for j in range(0, shape[1]):
            ri = m - i
            rj = n - j
            result[i, j] = kernel[ri, rj]
    return result


def get_slice(img, i, j, kernel_size, padding_size, output):
    """TODO: define a class to yield the (ki, kj) and (si, ei), (sj, ej)"""
    si = i - padding_size
    sj = j - padding_size
    ei = si + kernel_size
    ej = sj + kernel_size

    ki = 0
    kj = 0
    if si < 0:
        ki = -si
        si = 0
    if ei > img.shape[0]:
        ei = img.shape[0]

    if sj < 0:
        kj = -sj
        sj = 0
    if ej > img.shape[1]:
        ej = img.shape[1]

    output.fill(0.0)

    # print("i=(%s, %s), j=(%s, %s)" % (si, ei, sj, ej))
    kx = ki
    for x in range(si, ei):
        ky = kj
        for y in range(sj, ej):
            output[kx, ky] = img[x, y]
            ky += 1
        kx += 1

    return


def calc_conv(img, kernel, padding_size):
    """calculate 2d convoluation
    img: a 2D image of shape(height, width)
    kernel: a 2D filter of shape(ksize, ksize)
    stride: 1
    """
    patch = np.zeros(kernel.shape, dtype=np.float64)
    ksize = kernel.shape[0]

    shape = img.shape
    output = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            get_slice(img, i, j, ksize, padding_size, patch)
            patch *= kernel
            output[i, j] = np.sum(patch)
    return output


class Kernel(object):
    """a kernel over a 2D image of multiple channels.
    kernel size: self.size * self.size
    stride:      currently only one
    input: (in_channel, height, width) (CHW format)
           dose not support batch
    output: (1, height, width)
    """
    def __init__(self, size, activefunc, uuid):
        self.func = activefunc
        self.size = size
        self.bias = 0
        self.weights = None
        self.z = None
        self.output = None
        self.input_layer = None
        self.padding_size = int((size - 1)/2)

        self.delta = None
        self.delta_weights = None
        self.delta_bias = 0
        self.delta_input = None
        self.counter = 0
        self.uuid = uuid

        if size % 2 != 1:
            print("[WARNING] kernel size is not an even number: %s" % (size))
        return

    def init_weights(self):
        shape = self.input_layer.get_output().shape

        # 1. weights of the kernel
        # shape = (in_channel, kerenl_size, kernel_size)
        wshape = (shape[0], self.size, self.size)
        self.weights = init_nd_weights(wshape)
        self.delta_weights = np.zeros(self.weights.shape, dtype=np.float64)
        # print("[122] w.shape=%s, d.shape=%s" % (self.weights.shape, self.delta_weights.shape))

        # 2. z and output of current kernel
        self.z = np.zeros((shape[1], shape[2]), dtype=np.float64)
        self.output = np.zeros(self.z.shape, dtype=np.float64)
        self.delta = np.zeros(self.z.shape, dtype=np.float64)

        # 3. input_delta
        self.delta_input = np.zeros(shape, dtype=np.float64)
        return

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer
        input_data = input_layer.get_output()
        shape = input_data.shape
        if len(shape) != 3:
            print("wrong input layer shape: %s %s" % (input_layer, shape))
            return -1

        return 0

    def active(self):
        """compute one out-channel using current kernel."""
        self.z.fill(self.bias)

        # 1. calc z
        input_data = self.input_layer.get_output()
        shape = input_data.shape
        for i in range(shape[0]):
            channel = input_data[i]
            self.z += calc_conv(channel, self.weights, self.padding_size)

        # 2. activate
        self.func.forward(self.z, out=self.output)
        self.counter += 1
        return self.output

    def calc_weight_deltaX(self):
        """delta_weight = Delta(l+1) * X"""
        self.delta_weights.fill(0.0)
        patch = np.zeros((self.size, self.size))

        input_data = self.input_layer.get_output()
        # print("[163] input.shape=%s" % (input_data.shape,))

        shape = self.delta.shape
        for c in range(input_data.shape[0]):
            img = input_data[c]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if self.delta[i, j] == 0.0:
                        continue

                    # print("[173] img.shape=%s" % (img.shape,))
                    get_slice(img, i, j, self.size, self.padding_size, patch)
                    # print("[175] patch.shape=%s, delta.shape=%s" % (patch.shape, self.delta.shape))
                    # print("[177] delta_weights.shape=%s" % (self.delta_weights.shape,))
                    self.delta_weights[c] += (patch * self.delta[i, j])

        self.delta_bias = np.sum(self.delta)
        return

    def calc_weight_delta(self):
        """delta_weight = Delta(l+1) * X"""
        self.delta_weights.fill(0.0)
        patch = np.zeros((self.size, self.size))

        input_data = self.input_layer.get_output()
        shape = self.delta.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if self.delta[i, j] == 0.0:
                    continue
                for c in range(input_data.shape[0]):
                    img = input_data[c]
                    get_slice(img, i, j, self.size, self.padding_size, patch)
                    self.delta_weights += (patch * self.delta[i, j])

        self.delta_bias = np.sum(self.delta)

        if self.uuid == 3 and self.counter % 100 == 0:
            print("[200] w.delta=%s, bias.delta=%s" % (print_matrix_row(self.delta_weights), self.delta_bias))
            print("[206] delta=%s" % (np.sum(np.absolute(self.delta))))
        return

    def calc_input_delta(self):
        """(1) flip the kernel 180 degrees;
           (2) Calc convolution based on the output_delta;
        """
        # get the number of input channels
        channels = self.weights.shape[0]
        xweights = np.zeros(self.weights[0].shape, dtype=np.float64)
        for c in range(channels):
            flip180X(self.weights[c], xweights)
            self.delta_input[c] = calc_conv(self.delta, xweights, self.padding_size)
        return

    def calc_error(self, output_delta):
        """output_delta is the delta of the corresponding output-channel of current kernel."""
        # 0. calculate the derivative with z
        self.func.backward(self.z, self.output, out=self.delta)
        # print("[198] xx delta.shape=%s, output.shape=%s" % (self.delta.shape, output_delta.shape))
        self.delta *= output_delta

        # 1. calc kernel weight_delta
        self.calc_weight_delta()

        # 2. calc input_delta
        self.calc_input_delta()
        return

    def clear_delta(self):
        self.delta_weights.fill(0.0)
        self.delta_bias = 0.0
        # self.delta_input.fill(0.0)
        return

    def get_delta_input(self):
        return self.delta_input

    def update_weight(self, lr, l2):
        if l2 > 0:
            self.delta_weights += (self.weights * l2)

        self.weights -= (lr * self.delta_weights)
        self.bias -= (lr * self.delta_bias)
        self.clear_delta()
        return


class ConvLayer(Layer):
    def __init__(self, name):
        super(ConvLayer, self).__init__(name, 0)
        self.kernels = []
        self.delta = None
        return

    def add_kernel(self, kernel):
        self.kernels.append(kernel)
        self.size = len(self.kernels)
        return

    def add_kernels(self, kernels):
        self.kernels.extend(kernels)
        self.size = len(self.kernels)
        return

    def set_kernels(self, kernels):
        self.kernels = kernels
        self.size = len(self.kernels)
        return

    def init(self):
        # self.size = len(self.kernels)
        x = self.input_layer.get_output()
        shape = (self.size, x.shape[1], x.shape[2])
        self.output = np.zeros(shape, dtype=np.float64)
        self.delta = np.zeros(shape, dtype=np.float64)

        for i in range(len(self.kernels)):
            k = self.kernels[i]
            k.set_input_layer(self.input_layer)
            k.init_weights()
        return

    def active(self):
        x = self.input_layer.get_output()
        for i in range(len(self.kernels)):
            k = self.kernels[i]
            self.output[i, :, :] = k.active()
        return

    def calc_input_delta(self, delta):
        """delta: is the result of this function;
                  It is the delta if the input of current layer.
        """
        delta.fill(0.0)
        for i in range(len(self.kernels)):
            k = self.kernels[i]
            delta += k.get_delta_input()
        return

    def calc_error(self):
        self.next_layer.calc_input_delta(self.delta)
        for i in range(len(self.kernels)):
            k = self.kernels[i]
            k.calc_error(self.delta[i])
        return

    def update_weights(self, lr):
        for i in range(len(self.kernels)):
            self.kernels[i].update_weight(lr, self.lambda2)
        return


class MaxPoolingLayer(Layer):
    """None overlap max pooling layer."""
    def __init__(self, name, k1, k2):
        super(MaxPoolingLayer, self).__init__(name, k1*k2)
        self.k1 = k1
        self.k2 = k2
        self.transform_output_flag = False
        self.input_keeper = None
        self.channel_num = 0
        self.delta = None
        return

    def init(self):
        if self.input_layer is None:
            print("ERROR: input layer is None.")
            return

        if self.next_layer is None:
            print("ERROR: next layer is None.")
            return

        # 1. set a keeper matrix to keep track of the selected position
        x = self.input_layer.get_output()
        self.channel_num = x.shape[0]
        self.input_keeper = np.zeros(x.shape, dtype=np.int8)

        # 2. set output
        d1 = x.shape[1] // self.k1
        if d1 * self.k1 != x.shape[1]:
            d1 += 1
        d2 = x.shape[2] // self.k2
        if d2 * self.k2 != x.shape[2]:
            d2 += 1

        self.size = x.shape[0] * d1 * d2
        shape = (x.shape[0], d1, d2)
        self.output = np.zeros(shape)
        self.delta = np.zeros(shape)
        if type(self.next_layer) is ConvLayer:
            self.transform_output_flag = False
        else:
            self.transform_output_flag = True
        return

    def get_max(self, layer, i, j):
        bi = i * self.k1
        bj = j * self.k2

        ei = bi + self.k1;
        if ei > layer.shape[0]:
            ei = layer.shape[0]
        ej = bj + self.k2
        if ej > layer.shape[1]:
            ej = layer.shape[1]

        mi = bi
        mj = bj
        value = layer[mi, mj]
        for ki in range(bi, ei):
            for kj in range(bj, ej):
                if layer[ki, kj] > value:
                    value = layer[ki, kj]
                    mi = ki
                    mj = kj

        return mi, mj

    def active(self):
        self.input_keeper.fill(0)
        indata = self.input_layer.get_output()

        shape = self.output.shape
        for c in range(shape[0]):
            layer = indata[c, :, :]
            for i in range(shape[1]):
                for j in range(shape[2]):
                    mi, mj = self.get_max(layer, i, j)
                    self.output[c, i, j] = layer[mi, mj]
                    self.input_keeper[c, mi, mj] = 1
        return

    def calc_error(self):
        # 1. calc layer delta
        if self.transform_output_flag:
            tmp = np.zeros(self.delta.shape).reshape(-1)
            self.next_layer.calc_input_delta(tmp)
            np.copyto(self.delta, tmp.reshape(self.delta.shape, order="C"))
        else:
            self.next_layer.calc_input_delta(self.delta)

        # 2. since no weights, nor activation function
        #     no need to calc delta_weights
        return

    def calc_input_delta(self, input_delta):
        """calculate input_delta based on input_keeper and self.delta."""
        # 1. test channel nums are same
        if input_delta.shape[0] != self.delta.shape[0]:
            print("Bug, different channels: %s Vs. %s", input_delta.shape, self.delta.shape)

        # 2. calc input_delta
        input_delta.fill(0.0)
        shape = self.delta.shape
        for c in range(shape[0]):
            layer = self.input_keeper[c, :, :]
            for i in range(shape[1]):
                for j in range(shape[2]):
                    mi, mj = self.get_max(layer, i, j)
                    input_delta[c, mi, mj] = self.delta[c, i, j]
        return

    def update_weights(self, lr):
        """do nothing"""
        return

    def get_output(self):
        if self.transform_output_flag:
            return self.output.reshape(-1, order="C")
        return self.output


def print_matrix_row(m):
    t = m.reshape(-1)
    result = "["
    for e in t:
        result += ("%.4f, " % (e,))

    result += "]"
    return result
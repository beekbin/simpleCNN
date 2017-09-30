from __future__ import division
from __future__ import print_function
import numpy as np
import logging
from nn_layer import Layer
from conv_layer import ConvLayer


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

        ei = bi + self.k1
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
            msg = ("Bug, different channels: %s Vs. %s", input_delta.shape, self.delta.shape)
            logging.error(msg)
            print(msg)
            return

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

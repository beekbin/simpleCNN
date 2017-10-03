from __future__ import print_function

import numpy as np
import sys
import random
import logging

sys.path.insert(0, "../")
from nn.embedding_layer import EmbeddingLayer
from nn.nn_layer import InputLayer
from nn.nn_layer import ActiveLayer
from nn.simple_nn import NNetwork


class FakeOutputLayer(ActiveLayer):
    def __init__(self, name, size):
        super(FakeOutputLayer, self).__init__(name, size)
        return

    def calc_input_delta(self, output):
        np.copyto(output, self.delta)
        return

    def active(self):
        self.output = self.input_layer.get_output()
        return

    def calc_error(self, labels):
        # print("[30] labels=%s\noutput=%s\n" % (labels, self.output))
        self.delta = self.output - labels
        return

    def calc_cost(self, labels):
        cost = np.sum(np.absolute(self.delta))
        # print("[36] cost = %s" % cost)
        return cost


def get_random_vectors(m, n):
    result = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        result[i] = np.random.uniform(-1, 1, n)
    return result


def train_it(nn, y, data):
    random.shuffle(data)
    lr = 0.001
    for x in data:
        nn.train(x, y[x], lr)
    return


def evaluate_it(nn, y, data, prefix):
    total_cost = 0.0

    for x in data:
        _, cost = nn.evaluate(x, y[x])
        total_cost += cost

    logging.info("[%s] cost=%.3f" % (prefix, total_cost))
    return


def get_train_data(m):
    data = []
    for i in range(20 * m):
        x = i % m
        data.append(x)
    return data


def test1():
    m = 3
    n = 5
    y = get_random_vectors(m, n)

    nn = NNetwork()
    myin = InputLayer("input", 1)
    emb = EmbeddingLayer("emb1", m, n)
    myout = FakeOutputLayer("output", n)

    nn.set_input(myin)
    nn.set_output(myout)
    nn.add_hidden_layer(emb)
    nn.connect_layers()
    nn.set_log_interval(2000)

    print(emb.weights)
    print("*"*40)

    data = get_train_data(m)
    for i in range(500):
        train_it(nn, y, data)
        if i % 10 == 0:
            evaluate_it(nn, y, data, "epoch-%s" % i)

    print(y)
    print("*"*40)
    print(emb.weights)
    return


def main():
    test1()
    return 0


def setup_log():
    # logging.basicConfig(filename=logfile, format='[%(asctime)s] %(message)s', level=logging.DEBUG)
    logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] %(message)s', level=logging.DEBUG)
    return


if __name__ == "__main__":
    setup_log()
    sys.exit(main())

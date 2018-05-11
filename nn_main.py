from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import logging
from random import shuffle

from nn import nn_layer
from nn import activation
from nn import simple_nn
from util import mnist


def construct_nn(l2=0.0):
    img_input = nn_layer.InputLayer("mnist_input", 784)
    output_layer = nn_layer.SoftmaxOutputLayer("mnist_output", 10)

    # 1. set input and output layers
    nn = simple_nn.NNetwork()
    nn.set_input(img_input)
    nn.set_output(output_layer)

    # 2. add some hidden layers
    h1 = nn_layer.HiddenLayer("h1", 512, activation.tanhFunc)
    h1.set_lambda2(l2)
    nn.add_hidden_layer(h1)

    h2 = nn_layer.HiddenLayer("h2", 256, activation.reluFunc)
    h2.set_lambda2(l2)
    nn.add_hidden_layer(h2)

    h3 = nn_layer.HiddenLayer("h3", 256, activation.tanhFunc)
    h3.set_lambda2(l2)
    nn.add_hidden_layer(h3)

    h4 = nn_layer.HiddenLayer("h4", 10, activation.reluFunc)
    h4.set_lambda2(l2)
    nn.add_hidden_layer(h4)

    # 3. complete nn construction
    nn.connect_layers()
    logging.info("NN.info" + nn.get_detail())
    return nn


def train_it(nn, train_data, lr):
    labels = train_data[0]
    imgs = train_data[1]

    # shuffle the data
    alist = list(range(labels.shape[0]))
    shuffle(alist)

    for i in alist:
        label = labels[i, :]
        img = imgs[i, :]
        nn.train(img, label, lr)

    return


def evaluate_it(nn, test_data, prefix):
    labels = test_data[0]
    imgs = test_data[1]
    num = labels.shape[0]

    total_correct = 0
    total_cost = 0

    for i in range(num):
        label = labels[i, :]
        img = imgs[i, :]
        correct, cost = nn.evaluate(img, label)
        total_correct += correct
        total_cost += cost

    accuracy = float(total_correct) / num
    avg_cost = total_cost/num

    logging.info("[%s] accuracy=%.4f, avg_cost=%.4f" % (prefix, accuracy, avg_cost))
    return


def get_lr(step, current_lr):
    lrs = {0: 0.008, 1: 0.006, 4: 0.005, 5: 0.003, 6: 0.002, 8: 0.001, 10: 0.0005, 15: 0.0001}
    if step in lrs:
        return lrs[step]
    return current_lr


def train_nn(data_dir):
    nn = construct_nn()
    nn.set_log_interval(10000)
    train_data = mnist.load_data_1d(data_dir, "train")
    test_data = mnist.load_data_1d(data_dir, "test")
    if (train_data is None) or (test_data is None):
        msg = "[ERROR] failed to load data"
        print(msg)
        logging.error(msg)
        return

    lr = 0.005
    for i in range(100):
        lr = get_lr(i, lr)
        logging.info("begin epoch-%s, lr=%.6f" % (i, lr))
        train_it(nn, train_data, lr)
        evaluate_it(nn, test_data, "test")
        evaluate_it(nn, train_data, "train")
        logging.info("end epoch-%s" % (i,))

    return


def main():
    train_nn("./data/")
    return


def setup_log():
    logfile = "./train.%s.log" % (os.getpid())
    if len(sys.argv) > 1:
        logfile = sys.argv[1]
    print("logfile=%s" % (logfile,))
    logging.basicConfig(filename=logfile, format='[%(asctime)s] %(message)s', level=logging.DEBUG)
    return


if __name__ == "__main__":
    setup_log()
    sys.exit(main())

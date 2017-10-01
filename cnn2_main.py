from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import logging
import sys
import os
from random import shuffle
import numpy as np

from nn import nn_layer
from nn import conv_layer
from nn import pooling_layer
from nn import activation
from nn import simple_nn
from util import mnist


def get_kernels(n):
    result = []
    uuid = 1

    x = int(n/2)
    y = n - x

    # 1. 3x3 kernels
    for i in range(x):
        func = activation.reluFunc
        if i % 2 == 0:
            func = activation.tanhFunc
        kernel = conv_layer.Kernel(3, func, uuid)
        result.append(kernel)
        uuid += 1

    # 2. 5x5 kernels
    for i in range(y):
        func = activation.reluFunc
        if i % 2 == 0:
            func = activation.tanhFunc
        kernel = conv_layer.Kernel(5, func, uuid)
        result.append(kernel)
        uuid += 1
    return result


def construct_cnn(l2=0.0):
    img_input = nn_layer.InputLayer("mnist_input", 784)
    output_layer = nn_layer.SoftmaxOutputLayer("mnist_output", 10)

    # 1. set input and output layers
    nn = simple_nn.NNetwork()
    nn.set_input(img_input)
    nn.set_output(output_layer)

    # 2. add Conv-Pooling layers
    c1 = conv_layer.ConvLayer("conv1")
    c1.set_kernels(get_kernels(16))
    nn.add_hidden_layer(c1)

    # 2x2 none-overlapping max-pooling
    p1 = pooling_layer.MaxPoolingLayer("pool1", 2, 2)
    nn.add_hidden_layer(p1)

    # 3. add another Conv-Pooling layers
    c2 = conv_layer.ConvLayer("conv2")
    c2.set_kernels(get_kernels(32))
    nn.add_hidden_layer(c2)

    # 2x2 none-overlapping max-pooling
    p1 = pooling_layer.MaxPoolingLayer("pool2", 2, 2)
    nn.add_hidden_layer(p1)

    # 4. add some full-connected hidden layers
    h1 = nn_layer.HiddenLayer("h1", 512, activation.tanhFunc)
    h1.set_lambda2(l2)
    nn.add_hidden_layer(h1)

    h2 = nn_layer.HiddenLayer("h2", 128, activation.tanhFunc)
    h2.set_lambda2(l2)
    nn.add_hidden_layer(h2)

    h3 = nn_layer.HiddenLayer("h3", 10, activation.reluFunc)
    h3.set_lambda2(l2)
    nn.add_hidden_layer(h3)

    # 5. complete nn construction
    # print("%s" % (nn))
    fake_img = np.zeros((1, 28, 28))
    img_input.feed(fake_img)
    nn.connect_layers()
    logging.info("NN information:\n" + nn.get_detail())
    return nn


def train_it(nn, train_data, lr):
    labels = train_data[0]
    imgs = train_data[1]

    # shuffle the data
    alist = range(labels.shape[0])
    shuffle(alist)

    num = 0
    for i in alist:
        label = labels[i, :]
        img = imgs[i, :]
        nn.train(img, label, lr)
        if num % 1000 == 0:
            logging.info("num=%d" % num)
        num += 1

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

    msg = "[%s] accuracy=%.4f, avg_cost=%.4f" % (prefix, accuracy, avg_cost)
    logging.info(msg)
    return


def get_lr(step, current_lr):
    """simple learning rate scheduler"""
    lrs = {0: 0.008, 1: 0.005, 2: 0.003, 5: 0.002, 6: 0.001, 8: 0.0008, 10: 0.0005, 15: 0.0001}
    if step in lrs:
        return lrs[step]
    return current_lr


def train_nn(data_dir):
    nn = construct_cnn()
    # l2 = 1e-3
    # nn = construct_big_nn(l2)
    train_data = mnist.load_data_3d(data_dir, "train")
    test_data = mnist.load_data_3d(data_dir, "test")
    if (train_data is None) or (test_data is None):
        logging.error("[ERROR] failed to load data")
        return

    lr = 0.005
    for i in range(100):
        lr = get_lr(i, lr)
        msg = "begin epoch-%s, lr=%.6f" % (i, lr)
        logging.info(msg)
        train_it(nn, train_data, lr)
        evaluate_it(nn, test_data, "test")
        evaluate_it(nn, train_data, "train")
        logging.info("end epoch-%s" % (i, ))

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
    main()

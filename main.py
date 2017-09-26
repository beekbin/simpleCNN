from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from datetime import datetime
from random import shuffle
import numpy as np

from nn import nn_layer
from nn import conv_layer
from nn import activation
from nn import simple_nn
from util import mnist


def get_kernels():
    result = []
    uuid = 1

    # 1. 3x3 kernels
    for i in range(8):
        func = activation.reluFunc
        if i % 3 == 0:
            func = activation.tanhFunc
        kernel = conv_layer.Kernel(3, func, uuid)
        result.append(kernel)
        uuid += 1

    # 2. 5x5 kernels
    for i in range(8):
        func = activation.reluFunc
        if i % 3 == 0:
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
    c1.set_kernels(get_kernels())
    nn.add_hidden_layer(c1)

    # 2x2 none-overlapping max-pooling
    p1 = conv_layer.MaxPoolingLayer("pool1", 2, 2)
    nn.add_hidden_layer(p1)

    # 3. add some full-connected hidden layers
    h1 = nn_layer.HiddenLayer("h1", 512, activation.tanhFunc)
    h1.set_lambda2(l2)
    nn.add_hidden_layer(h1)

    h2 = nn_layer.HiddenLayer("h2", 128, activation.tanhFunc)
    h2.set_lambda2(l2)
    nn.add_hidden_layer(h2)

    h3 = nn_layer.HiddenLayer("h3", 10, activation.leakyReluFunc)
    h3.set_lambda2(l2)
    nn.add_hidden_layer(h3)

    # 3. complete nn construction
    # print("%s" % (nn))
    fake_img = np.zeros((1, 28, 28))
    img_input.feed(fake_img)
    nn.connect_layers()
    print(nn.get_detail())
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
        if num % 100 == 0:
            print("[%s] num=%d" % (str(datetime.now()), num))
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

    print("[%s][%s] accuracy=%.4f, avg_cost=%.4f" % (str(datetime.now()), prefix, accuracy, avg_cost))
    return


def get_lr(step, current_lr):
    lrs = {0: 0.008, 1: 0.006, 4: 0.005, 5: 0.003, 6: 0.002, 8: 0.001, 10: 0.0005, 15: 0.0001}
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
        print("[ERROR] failed to load data")
        return

    lr = 0.005
    for i in range(100):
        lr = get_lr(i, lr)
        print("[%s] begin epoch-%s, lr=%.6f" % (str(datetime.now()), i, lr))
        train_it(nn, train_data, lr)
        evaluate_it(nn, test_data, "test")
        evaluate_it(nn, train_data, "train")
        print("[%s] end epoch-%s" % (str(datetime.now()), i))

    return


def main():
    train_nn("./data/")
    return


if __name__ == "__main__":
    main()

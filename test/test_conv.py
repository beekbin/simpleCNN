from __future__ import division
from __future__ import print_function
import sys
from multiprocessing import Process, Queue
from datetime import datetime
import numpy as np

sys.path.insert(0, "../")
from nn.conv_layer import ConvLayer
from nn.conv_layer import Kernel
from nn.nn_layer import InputLayer
from nn.nn_layer import HiddenLayer
from nn import activation


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
    output = np.zeros(shape, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            get_slice(img, i, j, ksize, padding_size, patch)
            patch *= kernel
            output[i, j] = np.sum(patch)
    return output


def calc_conv2(img, kernel, psize, output=None):
    patch = np.zeros(kernel.shape, dtype=np.float64)
    ksize = kernel.shape[0]

    shape = img.shape
    if output is None:
        output = np.zeros(shape, dtype=np.float64)

    si = 0
    di = psize + 1
    ei = ksize - psize - 1
    for i in range(shape[0]):
        if di <= 0:
            si += 1
        else:
            di -= 1

        if ei < shape[0]:
            ei += 1

        sj = 0
        dj = psize + 1
        ej = ksize - psize - 1
        for j in range(shape[1]):
            if dj <= 0:
                sj += 1
            else:
                dj -= 1
            if ej < shape[1]:
                ej += 1

            patch.fill(0.0)
            # print("i=(%s,%s, %s), j=(%s,%s, %s)" % (si, ei, di, sj, ej, dj))
            ddi = ei - si + di
            ddj = ej - sj + dj
            patch[di:ddi, dj:ddj] = img[si:ei, sj:ej]
            patch *= kernel
            output[i, j] = np.sum(patch)

    return output


def test():
    shape = (9, 9)
    kshape = (5, 5)
    x = np.array(range(81), dtype=np.float64).reshape(shape)
    kernel = np.ones(kshape, dtype=np.float64)
    padding_size = int((kshape[0]-1)/2)

    o = calc_conv(x, kernel, padding_size)
    oo = np.zeros(x.shape, dtype=np.float64)
    #oo = calc_conv2(x, kernel, padding_size)
    calc_conv2(x, kernel, padding_size, output=oo)
    print(x)
    print("*"*30)
    print(o)
    print("*"*30)
    print(oo)
    print("*"*30)
    print(oo-o)
    return


def test2():
    # 1. init
    shape = (28, 28)
    kshape = (3, 3)
    x = np.array(range(784), dtype=np.float64).reshape(shape)
    kernel = np.ones(kshape, dtype=np.float64)
    padding_size = int((kshape[0]-1)/2)

    # 2. test 1
    begin = datetime.now()
    for i in range(1000):
        o = calc_conv(x, kernel, padding_size)
    delta = datetime.now() - begin
    print("[%s] delta=%s" % (str(begin), str(delta)))

    # 3. test2
    begin = datetime.now()
    for j in range(1000):
        oo = calc_conv2(x, kernel, padding_size)
    delta = datetime.now() - begin
    print("[%s] delta=%s" % (str(begin), str(delta)))

    # 4. test3
    begin = datetime.now()
    oo = np.zeros(x.shape, dtype=np.float64)
    for j in range(1000):
        calc_conv2(x, kernel, padding_size, output=oo)
    delta = datetime.now() - begin
    print("[%s] delta=%s" % (str(begin), str(delta)))

    return


def test3():
    # 1. init
    shape = (28, 28)
    kshape = (3, 3)
    x = np.array(range(784), dtype=np.float64).reshape(shape)
    kernel = np.ones(kshape, dtype=np.float64)
    padding_size = int((kshape[0]-1)/2)

    # 2. test 1
    begin = datetime.now()
    for i in range(1000):
        o = calc_conv(x, kernel, padding_size)
    delta = datetime.now() - begin
    print("[%s] delta.1=%s" % (str(begin), str(delta)))

    # 3. test2
    begin = datetime.now()
    for i in range(1000):
        oo = calc_conv2(x, kernel, padding_size)
    delta = datetime.now() - begin
    print("[%s] delta.2=%s" % (str(begin), str(delta)))

    # 4. test3
    begin = datetime.now()
    oo = np.zeros(x.shape, dtype=np.float64)
    q = Queue()
    for i in range(1000):
        workers = []
        for j in range(10):
            myarg = (x, kernel, padding_size)
            t = Process(target=calc_conv2, args=myarg)
            workers.append(t)
            t.start()
        for t in workers:
            t.join()
    delta = datetime.now() - begin
    print("[%s] delta.3=%s" % (str(begin), str(delta)))

    return



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

    begin = datetime.now()
    print("[%s] begin to active." % (str(begin)))
    for i in range(1000):
        h1.active()
    end = datetime.now()
    delta = end - begin
    print("[%s] end of activation, delta=%s." % (str(end), str(delta)))
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

    begin = datetime.now()
    print("[%s] begin to active." % (str(begin)))
    for i in range(1000):
        conv.active()
    end = datetime.now()
    delta = end - begin
    print("[%s] end of activation, delta=%s." % (str(end), str(delta)))
    return


def main():
    test()
    # test2()
    test3()
    # test_normal_active()
    # test_active()
    return


if __name__ == "__main__":
    sys.exit(main())

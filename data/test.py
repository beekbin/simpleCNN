import os
import sys
import struct
import numpy as np


def load_data(path, dtype):
    if dtype == "train":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dtype == "test":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        print("[ERROR] dtype must be 'test' or 'train' vs. %s" % (dtype))
        return None

    # 1. load labels
    with open(fname_lbl, 'rb') as flbl:
        magic1, num1 = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)
        print("%s: magic=%d, number=%d" % (fname_lbl, magic1, num1))
        print(labels.shape)

    # 2. load images
    with open(fname_img, 'rb') as fimg:
        magic2, num2, rows, cols = struct.unpack(">IIII", fimg.read(16))
        imgs = np.fromfile(fimg, dtype=np.uint8).reshape(-1, rows * cols)
        print("%s: magic=%d, num=%d, rows=%d, cols=%d" % (fname_img, magic2, num2, rows, cols))
        print(imgs.shape)

    assert(num1 == num2)
    assert(rows == 28)
    assert(cols == 28)

    return


def main(argv):
    dtype = "test"
    load_data(".", dtype)
    dtype = "train"
    load_data(".", dtype)
    return


if __name__ == "__main__":
    sys.exit(main(sys.argv))

from __future__ import print_function
from __future__ import division

import os
import struct
import numpy as np


def normalize_img(imgs):
    result = imgs.astype(float)

    # convert data from [0, 255] to [0, 1.0]
    result /= 255.0
    avg = np.average(result, axis=1).reshape((-1, 1))
    result -= avg
    return result


def l2_normalize_img(imgs):
    result = imgs.astype(float)

    l2 = np.sqrt((result * result).sum(axis=1))
    result /= l2.resahpe((-1, 1))
    return result


def load_data(path, dset):
    if dset == "train":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dset == "test":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        print("[ERROR] data.set must be 'test' or 'train' vs. %s" % (dset))
        return None

    # 1. load labels
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)
        print("%s: magic=%d, number=%d" % (fname_lbl, magic, num))
        print(labels.shape)

    # 2. load images
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        imgs = np.fromfile(fimg, dtype=np.uint8).reshape(-1, rows*cols)
        print("%s: magic=%d, num=%d, rows=%d, cols=%d" % (fname_img, magic, num, rows, cols))

    imgs = normalize_img(imgs)
    labels = transform_label(labels, 10)
    return labels, imgs, rows, cols


def load_data_3d(path, dset):
    """load img in 3D matrix: (channel, Height, Width)."""
    labels, imgs, rows, cols = load_data(path, dset)
    imgs = imgs.reshape(-1, 1, rows, cols)
    print("images.shape=%s, labels.shape=%s" % (imgs.shape, labels.shape))
    return labels, imgs


def load_data_1d(path, dset):
    """load img in 1D vector. """
    labels, imgs, _, _ = load_data(path, dset)
    print("images.shape=%s, labels.shape=%s" % (imgs.shape, labels.shape))
    return labels, imgs


def transform_label(label_1d, d2):
    num = label_1d.shape[0]
    result = np.zeros((num, d2))

    for i in range(num):
        label = label_1d[i]
        result[i, label] = 1
    return result

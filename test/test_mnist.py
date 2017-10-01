from __future__ import print_function

import sys
sys.path.insert(0, "../")
from util import mnist


def main():
    data = "../data/"

    train_data = mnist.load_data_3d(data, "train")
    test_data = mnist.load_data_3d(data, "test")

    train_data = mnist.load_data_1d(data, "train")
    test_data = mnist.load_data_1d(data, "test")
    return


if __name__ == "__main__":
    sys.exit(main())

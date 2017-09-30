#!/bin/bash

files="
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
"

url="wget http://yann.lecun.com/exdb/mnist/"

for fname in $files ; do
    wget $url$fname
    gunzip -k $fname
done


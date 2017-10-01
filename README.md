# simpleCNN
[A convolution layer](https://github.com/beekbin/simpleCNN/blob/master/nn/conv_layer.py) and [a max pooling layer](https://github.com/beekbin/simpleCNN/blob/master/nn/pooling_layer.py) are added to my [vanilla neural network framework](https://github.com/beekbin/SimpleNN). 

With these two additional layers, a CNN can be built via this simple framework.  The `main.py` file demonstrates how to use the simple framework to build a CNN, and how to train the CNN with `MNIST` dataset.

# Construct the CNN
This neural networks contain 1 convolution layer, 1 max pooling layer, 3 fully connected hidden layers
and a softmax output layer. 

The convolution layer has 16 kernels, 8 of them are `3x3` kernels, and 8 of them are `5x5` kernels. With zero-padding, each kernel will preserve the dimensions of the input data.
The max pooling layer does `2x2` none-overlapping max pooling, and the dimensions of its output will be half (or half+1) of the input dimensions.


```python
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
    p1 = pooling_layer.MaxPoolingLayer("pool1", 2, 2)
    nn.add_hidden_layer(p1)

    # 3. add some full-connected hidden layers
    h1 = nn_layer.HiddenLayer("h1", 512, activation.tanhFunc)
    h1.set_lambda2(l2)
    nn.add_hidden_layer(h1)

    h2 = nn_layer.HiddenLayer("h2", 128, activation.tanhFunc)
    h2.set_lambda2(l2)
    nn.add_hidden_layer(h2)

    h3 = nn_layer.HiddenLayer("h3", 10, activation.reluFunc)
    h3.set_lambda2(l2)
    nn.add_hidden_layer(h3)

    # 3. complete nn construction
    # print("%s" % (nn))
    fake_img = np.zeros((1, 28, 28))
    img_input.feed(fake_img)
    nn.connect_layers()
    print(nn.get_detail())
    return nn
```



# Run it
### 1. get data
```bash
cd simpleCNN
cd data
sh get.sh
```

### 2. train the model
```bash
cd simpleCNN
python main.py
```

Because of the convolution layer, the training process is very slow: takes around 4 hours to finish one echo. But the result is promising: after the training of the second epoch, it can get 98.50% correctness on testing set.
```console
[2017-09-30 14:43:45.445192][test] accuracy=0.9741, avg_cost=0.0796
[2017-09-30 16:17:21.345727][train] accuracy=0.9792, avg_cost=0.0673
[2017-09-30 20:55:07.247804][test] accuracy=0.9854, avg_cost=0.0480
[2017-09-30 22:28:31.628410][train] accuracy=0.9890, avg_cost=0.0345
```

As a comparison, [the similar simple NN model (without the ConvLayer + MaxPoolingLayer)](https://github.com/beekbin/SimpleNN) gets 94.74% correctness on testing set after the first epoch, and only gets 98.11% at best (11 epochs).
```console
[2017-09-24 23:12:26.834471][train] accuracy=0.9526, avg_cost=0.1555
[2017-09-24 23:12:27.730683][test] accuracy=0.9474, avg_cost=0.1725
```

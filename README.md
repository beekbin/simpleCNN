# simpleCNN

# Construct the CNN
This neural networks contain 1 convolution layer, 1 max pooling layer, 3 fully connected hidden layers
and a softmax output layer. 

The convolution layer has 16 kernels, 8 of them are 3x3 kernels, and 8 of them are 5x5 kernels. With zero-padding, each kernel will preserve the dimensions of the input data.
The max pooling layer does 2x2 none-overlapping max pooling, and the dimensions of its output will be half (or half+1) of the input dimensions.


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
## 1. get data
```bash
cd simpleCNN
cd data
sh get.sh
```

## 2. train the model
```bash
cd simpleCNN
python main.py
```

It is very slow. After the training of the forth epoch, it can get 98.50% correctness on testing set.
```console
[2017-09-27 14:30:37.731326][test] accuracy=0.9670, avg_cost=0.1050
[2017-09-27 16:27:47.301980][train] accuracy=0.9727, avg_cost=0.0874
[2017-09-27 21:26:10.912253][test] accuracy=0.9740, avg_cost=0.0865
[2017-09-27 23:06:15.987296][train] accuracy=0.9798, avg_cost=0.0620
[2017-09-28 13:11:32.286327][test] accuracy=0.9836, avg_cost=0.0509
[2017-09-28 14:54:31.384333][train] accuracy=0.9915, avg_cost=0.0277
[2017-09-28 20:06:15.679026][test] accuracy=0.9850, avg_cost=0.0481
[2017-09-28 22:04:24.954797][train] accuracy=0.9947, avg_cost=0.0176
```

As a comparison, [the similar simple NN model](https://github.com/beekbin/SimpleNN) gets 94.47% correctness on testing set after the first epoch, and only gets 98.11% at best.
```console
[2017-09-24 23:12:26.834471][train] accuracy=0.9526, avg_cost=0.1555
[2017-09-24 23:12:27.730683][test] accuracy=0.9474, avg_cost=0.1725
```

# simpleCNN

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

It is very slow. After the training of the second epoch, it can get 97.40% correctness on testing set.
```console
[2017-09-27 14:30:37.731326][test] accuracy=0.9670, avg_cost=0.1050
[2017-09-27 16:27:47.301980][train] accuracy=0.9727, avg_cost=0.0874
[2017-09-27 21:26:10.912253][test] accuracy=0.9740, avg_cost=0.0865
[2017-09-27 23:06:15.987296][train] accuracy=0.9798, avg_cost=0.0620
```

As a comparison, [the similar simple NN model](https://github.com/beekbin/SimpleNN) gets 94.47% correctness on testing set after the first epoch.
```console
[2017-09-24 23:12:26.834471][train] accuracy=0.9526, avg_cost=0.1555
[2017-09-24 23:12:27.730683][test] accuracy=0.9474, avg_cost=0.1725
```

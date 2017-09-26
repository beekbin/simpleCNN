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

It is very slow, taking around 4 hours for one epoch. After the training of second epoch, it can get 96.81% correctness on testing set. 
```console
[2017-09-24 23:09:24.113187][train] accuracy=0.9534, avg_cost=0.1490
[2017-09-24 23:27:59.361241][test] accuracy=0.9502, avg_cost=0.1635
[2017-09-25 21:45:17.588531][train] accuracy=0.9757, avg_cost=0.0800
[2017-09-25 22:04:05.682715][test] accuracy=0.9681, avg_cost=0.1066
```

As a comparison, [the similar simple NN model](https://github.com/beekbin/SimpleNN) gets 94.47% correctness on testing set after the first epoch.
```console
[2017-09-24 23:12:26.834471][train] accuracy=0.9526, avg_cost=0.1555
[2017-09-24 23:12:27.730683][test] accuracy=0.9474, avg_cost=0.1725
```

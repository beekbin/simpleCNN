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

It is very slow. After the training of first epoch, it can get 95.34% correctness on training set.
```console
[2017-09-24 23:09:24.113187][train] accuracy=0.9534, avg_cost=0.1490
[2017-09-24 23:27:59.361241][test] accuracy=0.9502, avg_cost=0.1635
```

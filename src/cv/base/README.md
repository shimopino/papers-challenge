# Clone of "Pytorch Official ImageNet Example"

## dataset

```sh
$ bash download.sh tiny-imagenet-200
```

## train

> When training a model in a GPU environment, an error will occur if the number of workers of DataLoader is set to 0.

```sh
$ python main.py --data data/tiny-imagenet-200 --workers 0 --gpu 0
```
# Clone of "Pytorch Official ImageNet Example"

## dataset

```sh
$ bash download.sh tiny-imagenet-200
```

## train

```sh
$ python main.py --data data/tiny-imagenet-200 --workers 8 --gpu 0
```

> When training a model in a GPU environment, an error will occur if the number of workers of DataLoader is set to 0.

- [Runtime Error with DataLoader: exited unexpectedly](https://github.com/pytorch/pytorch/issues/5301)
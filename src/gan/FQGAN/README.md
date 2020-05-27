# Feature Quantization Improves GAN Training, 2020

### download

```bash
# add celeba dataset to ../data/celeba
$ bash download celeba
```

### start training

```bash
$ python main.py

# if you don't want to create __pycache__
$ python -B main.py
```

### start tensorboard

```bash
$ tensorboard --logdir logs --host 0.0.0.0
```
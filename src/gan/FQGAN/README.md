# Feature Quantization Improves GAN Training, 2020

[[arXiv:2004.02088] Feature Quantization Improves GAN Training](https://arxiv.org/abs/2004.02088)

## Dependencies

- Ubuntu 18.04+ (any Linux Distribution which supports NVIDIA Docker)
- VSCode (which supports "[Remote Containers](https://github.com/microsoft/vscode-dev-containers)")
- GPU Environment (my case is "RTX 2080ti")

When you start this project, please select VSCode Command `Open Folder in Container`.

## Configuration

When you test Feature Quantization, you change some properties in `./config/config.py`.

```python
# vq_type: [None, "Normal", "EMA"]
vq_type: str = None

# dictionary size: [1, ..., 10] 
# in the paper, they use 10
# if you select 1, it means binary quantization.
dict_size: int = 1

# [2, 3, 4] layer where you want to apply Feature Quantization
quant_layers: List = field(default_factory=lambda: [3])
```

## start training

```bash
$ python main.py

# if you don't want to create __pycache__
$ python -B main.py
```

if you use multirun on Hydra

```sh
$ python main.py --multirun model.FQ.vq_type=Normal,EMA model.FQ.dict_size=1,5,10 hydra.sweep.dir=\${logging.log_dir} hydra.sweep.subdir=\${model.FQ.vq_type}/\${model.FQ.dict_size} -m
```

## start tensorboard

```bash
$ tensorboard --logdir logs --host 0.0.0.0

# when "tensorboard: command not found" error occured, use command below.
$ python /home/devpapers/.local/lib/python3.7/site-packages/tensorboard/main.py --logdir logs --host=0.0.0.0
```

## Results

CIFAR10 (32x32)

|model|FQ Type|Dictionary Size|Layers|FID|IS|KID|
|---|---|---|---|---|---|---|
|SNGAN|-|-|-|16.35|7.89|0.0123|
|SNGAN-pd|-|-|-|11.84|8.44|0.0073|
|FQGAN|-|-|-|-|-|-|


### Citations

FQGAN

```
@misc{zhao2020feature,
    title={Feature Quantization Improves GAN Training},
    author={Yang Zhao and Chunyuan Li and Ping Yu and Jianfeng Gao and Changyou Chen},
    year={2020},
    eprint={2004.02088},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

This project heavily relies on mimicry.

```
@article{lee2020mimicry,
    title={Mimicry: Towards the Reproducibility of GAN Research},
    author={Kwot Sin Lee and Christopher Town},
    booktitle={CVPR Workshop on AI for Content Creation},
    year={2020},
}
```
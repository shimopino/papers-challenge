from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # dataset
    dataset_dir: str = "./datasets"
    dataset_name: str = "cifar10"

    # model SNGAN
    nz: int = 128
    ngf: int = 256
    ndf: int = 128
    loss_type: str = "hinge"  # "hinge", "ns", "gan", "wasserstein"

    # model FQGAN
    vq_type: str = None  # "Normal"
    dict_size: int = 1
    quant_layers: List = field(default_factory=lambda: [3])

    # optimizer
    lrD: float = 2e-4
    lrG: float = 2e-4
    beta1: float = 0.0
    beta2: float = 0.9

    # training
    num_workers: int = 4
    batch_size: int = 64
    n_dis: int = 5
    num_steps: int = 100000
    is_amp: bool = True
    opt_level: str = "O1"

    # evaluation
    # metric: str = 'fid'
    num_real_samples: int = 50000
    num_fake_samples: int = 50000
    evaluate_step: int = 100000

    # metric: str = 'kid'
    num_subsets: int = 50
    subset_size: int = 1000
    evaluate_step: int = 100000

    # metric: str = 'inception_score'
    num_samples: int = 50000
    evaluate_step: int = 100000

    # logging
    log_dir: str = "./logs/sngan_32_128z_256ngf_128ndf"

    @property
    def betas(self):
        return (self.beta1, self.beta2)

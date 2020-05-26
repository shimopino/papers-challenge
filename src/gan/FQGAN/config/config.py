from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Config:
    # general settings
    seed: int = 42
    ngpu: int = 1
    amp: bool = True
    opt_level: str = "O1"

    # dataset
    datapath: str = "../data/celeba"
    num_workers: int = 2
    image_size: int = 64

    # model
    nc: int = 3
    nz: int = 128
    ngf: int = 768
    ndf: int = 768
    bottom_width: int = 4
    use_sn: bool = True
    use_cbm: bool = True
    use_vq: bool = False
    use_ema_vq: bool = False
    dict_size: int = 5  # 2^10 --> 1024
    quant_layers: List = field(default_factory=lambda: [2, 3, 4])

    # training
    n_epochs: int = 30
    batch_size: int = 128
    beta1: float = 0.0
    beta2: float = 0.9
    lrD: float = 1e-4
    lrG: float = 1e-4
    is_ortho: bool = False
    factor: float = 1e-4
    loss_type: str = "bce"

    # logging
    log_dir: str = "logs"
    flush_secs: int = 5
    print_iters: int = 200

    @property
    def betas(self):
        return (self.beta1, self.beta2)

    @property
    def is_cuda(self):
        return self.ngpu > 0 and torch.cuda.is_available()

    @property
    def device(self):
        return torch.device("cuda:0" if self.is_cuda else "cpu")

    def __repr__(self):
        msg = "\n"
        for key, value in self.__dict__.items():
            msg += f"{key}: {value}\n"
        return msg

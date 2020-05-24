import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from training.logger import Logger
from training.utils import set_seed, count_parameters_float32
from datasets import data_utils
from models.FQGAN import ResNetGenerator, ResNetDiscriminator


def main(cfg):

    # logger = Logger(cfg.log_dir, cfg.flush_secs)

    if cfg is not None:
        print(f"configuration: {cfg}")

    if cfg.seed:
        set_seed(cfg.seed)
        print(f"{cfg.seed}")

    print(f"device setting: {cfg.device}")

    dataset = data_utils.get_celeba_dataset(
        root=cfg.datapath, image_size=cfg.image_size
    )
    print(len(dataset))

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True
    )

    real_batch = next(iter(dataloader))
    print(f"sample real batch size: {real_batch[0].size()}")
    # logger.add_image("sample", real_batch[0])

    netG = ResNetGenerator(cfg.nz, cfg.ngf, cfg.nc, cfg.bottom_width, cfg.use_sn).to(
        cfg.device
    )
    print(f"Generator Memory: {count_parameters_float32(netG):.2f} MB")

    netD = ResNetDiscriminator(
        cfg.nc, cfg.ndf, cfg.use_sn, cfg.use_vq, cfg.dict_size
    ).to(cfg.device)
    print(f"Discriminator Memory: {count_parameters_float32(netD):.2f} MB")

    if (cfg.is_cuda) and (cfg.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(cfg.ngpu)))
        netD = nn.DataParallel(netD, list(range(cfg.ngpu)))


if __name__ == "__main__":
    cfg = Config()
    main(cfg)

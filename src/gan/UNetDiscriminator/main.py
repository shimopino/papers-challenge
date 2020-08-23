# import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
import torch_mimicry as mmc
from apex import amp
from models.unetgan_32 import UNetGenerator, UNetDiscriminator


def get_models(cfg):

    # netG = UNetGenerator(nz=cfg.model.GAN.nz,
    #                      ngf=cfg.model.GAN.ngf,
    #                      loss_type=cfg.model.GAN.loss_type,
    #                      is_amp=cfg.training.is_amp)

    # netD = UNetDiscriminator(ndf=cfg.model.GAN.ndf,
    #                          loss_type=cfg.model.GAN.loss_type,
    #                          is_amp=cfg.training.is_amp)

    netG = UNetGenerator(nz=128,
                         ngf=256,
                         loss_type="ns",
                         is_amp=False)

    netD = UNetDiscriminator(ndf=128,
                             loss_type="ns",
                             is_amp=False)

    return netG, netD


def build_models(cfg, device):
    # Define models and optimizers
    netG, netD = get_models(cfg)
    netG, netD = netG.to(device), netD.to(device)

    # optD = optim.Adam(netD.parameters(),
    #                   cfg.optimizer.lrD,
    #                   betas=(cfg.optimizer.beta1, cfg.optimizer.beta2))
    # optG = optim.Adam(netG.parameters(),
    #                   cfg.optimizer.lrG,
    #                   betas=(cfg.optimizer.beta1, cfg.optimizer.beta2))
    optD = optim.Adam(netD.parameters(),
                      0.0002,
                      betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(),
                      0.0002,
                      betas=(0.0, 0.9))

    # if cfg.training.is_amp:
    #     [netG, netD], [optG, optD] = amp.initialize([netG, netD],
    #                                                 [optG, optD],
    #                                                 opt_level=cfg.training.opt_level)
    if True:
        [netG, netD], [optG, optD] = amp.initialize([netG, netD],
                                                    [optG, optD],
                                                    opt_level="O1")

    return netG, netD, optG, optD


# @hydra.main(config_path="config.yaml")
# def main(cfg: DictConfig):
def main():
    # get original path
    # cwd = hydra.utils.get_original_cwd()
    # DATASET_DIR = cwd + "/" + cfg.dataset.dataset_dir

    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    # dataset = mmc.datasets.load_dataset(root=DATASET_DIR,
    #                                     name=cfg.dataset.dataset_name)
    dataset = mmc.datasets.load_dataset(root="dataset",
                                        name="cifar10")
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=cfg.training.batch_size,
    #     shuffle=True,
    #     num_workers=cfg.training.num_workers)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4)

    # Define models and optimizers
    netG, netD, optG, optD = build_models(None, device)

    #########################
    #        Training
    #########################
    # Start training
    # trainer = mmc.training.Trainer(
    #     netD=netD,
    #     netG=netG,
    #     optD=optD,
    #     optG=optG,
    #     n_dis=cfg.training.n_dis,
    #     num_steps=cfg.training.num_steps,
    #     dataloader=dataloader,
    #     log_dir=cfg.logging.log_dir,
    #     device=device)
    # trainer.train()
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=5,
        num_steps=100000,
        dataloader=dataloader,
        log_dir="logs",
        device=device)
    trainer.train()

    ##########################
    #       Evaluation
    ##########################
    # Evaluate fid
    # mmc.metrics.evaluate(
    #     metric="fid",
    #     log_dir=cfg.logging.log_dir,
    #     netG=netG,
    #     dataset_name=cfg.dataset.dataset_name,
    #     num_real_samples=cfg.metrics.fid.num_real_samples,
    #     num_fake_samples=cfg.metrics.fid.num_fake_samples,
    #     evaluate_step=cfg.metrics.fid.evaluate_step,
    #     device=device)
    mmc.metrics.evaluate(
        metric="fid",
        log_dir="logs",
        netG=netG,
        dataset_name="cifar10",
        num_real_samples=50000,
        num_fake_samples=50000,
        evaluate_step=100000,
        device=device)

    # Evaluate inception score
    # mmc.metrics.evaluate(
    #     metric='inception_score',
    #     log_dir=cfg.logging.log_dir,
    #     netG=netG,
    #     num_samples=cfg.metrics.inception_score.num_samples,
    #     evaluate_step=cfg.metrics.inception_score.evaluate_step,
    #     device=device)
    mmc.metrics.evaluate(
        metric='inception_score',
        log_dir="logs",
        netG=netG,
        num_samples=50000,
        evaluate_step=100000,
        device=device)

    # Evaluate kid
    # mmc.metrics.evaluate(
    #     metric='kid',
    #     log_dir=cfg.logging.log_dir,
    #     netG=netG,
    #     dataset_name=cfg.dataset.dataset_name,
    #     num_subsets=cfg.metrics.kid.num_subsets,
    #     subset_size=cfg.metrics.kid.subset_size,
    #     evaluate_step=cfg.metrics.kid.evaluate_step,
    #     device=device)
    mmc.metrics.evaluate(
        metric='kid',
        log_dir="logs",
        netG=netG,
        dataset_name="cifar10",
        num_subsets=50,
        subset_size=1000,
        evaluate_step=100000,
        device=device)


if __name__ == "__main__":
    main()

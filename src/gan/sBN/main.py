import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
import torch_mimicry as mmc
from apex import amp
from models.sbn_sngan_32 import SBNSNGANGenerator32, SBNSNGANDiscriminator32


def get_models(cfg):

    netG = SBNSNGANGenerator32(nz=cfg.model.GAN.nz,
                               ngf=cfg.model.GAN.ngf,
                               loss_type=cfg.model.GAN.loss_type,
                               is_amp=cfg.training.is_amp)
    netD = SBNSNGANDiscriminator32(ndf=cfg.model.GAN.ndf,
                                   loss_type=cfg.model.GAN.loss_type,
                                   is_amp=cfg.training.is_amp)

    return netG, netD


def build_models(cfg, device):
    # Define models and optimizers
    netG, netD = get_models(cfg)
    netG, netD = netG.to(device), netD.to(device)

    optD = optim.Adam(netD.parameters(),
                      cfg.optimizer.lrD,
                      betas=(cfg.optimizer.beta1, cfg.optimizer.beta2))
    optG = optim.Adam(netG.parameters(),
                      cfg.optimizer.lrG,
                      betas=(cfg.optimizer.beta1, cfg.optimizer.beta2))

    if cfg.training.is_amp:
        [netG, netD], [optG, optD] = amp.initialize([netG, netD],
                                                    [optG, optD],
                                                    opt_level=cfg.training.opt_level)

    return netG, netD, optG, optD


@hydra.main(config_path="config.yaml")
def main(cfg: DictConfig):
    # get original path
    cwd = hydra.utils.get_original_cwd()
    DATASET_DIR = cwd + "/" + cfg.dataset.dataset_dir

    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root=DATASET_DIR,
                                        name=cfg.dataset.dataset_name)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers)

    # Define models and optimizers
    netG, netD, optG, optD = build_models(cfg, device)

    # Start training
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=cfg.training.n_dis,
        num_steps=cfg.training.num_steps,
        dataloader=dataloader,
        log_dir=cfg.logging.log_dir,
        device=device)
    trainer.train()

    # Evaluate fid
    mmc.metrics.evaluate(
        metric="fid",
        log_dir=cfg.logging.log_dir,
        netG=netG,
        dataset_name=cfg.dataset.dataset_name,
        num_real_samples=cfg.metrics.fid.num_real_samples,
        num_fake_samples=cfg.metrics.fid.num_fake_samples,
        evaluate_step=cfg.metrics.fid.evaluate_step,
        device=device)

    # Evaluate inception score
    mmc.metrics.evaluate(
        metric='inception_score',
        log_dir=cfg.logging.log_dir,
        netG=netG,
        num_samples=cfg.metrics.inception_score.num_samples,
        evaluate_step=cfg.metrics.inception_score.evaluate_step,
        device=device)

    # Evaluate kid
    mmc.metrics.evaluate(
        metric='kid',
        log_dir=cfg.logging.log_dir,
        netG=netG,
        dataset_name=cfg.dataset.dataset_name,
        num_subsets=cfg.metrics.kid.num_subsets,
        subset_size=cfg.metrics.kid.subset_size,
        evaluate_step=cfg.metrics.kid.evaluate_step,
        device=device)


if __name__ == "__main__":
    main()
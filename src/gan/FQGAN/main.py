from config import Config
import torch
import torch.optim as optim
import torch_mimicry as mmc
from apex import amp
from models.fqgan_32 import FQGANGenerator, FQGANDiscriminator


def build_models(cfg, device):
    # Define models and optimizers
    netG = FQGANGenerator(nz=cfg.nz,
                          ngf=cfg.ngf,
                          loss_type=cfg.loss_type,
                          is_amp=cfg.is_amp).to(device)
    netD = FQGANDiscriminator(ndf=cfg.ndf,
                              loss_type=cfg.loss_type,
                              vq_type=cfg.vq_type,
                              dict_size=cfg.dict_size,
                              quant_layers=cfg.quant_layers,
                              is_amp=cfg.is_amp).to(device)
    optD = optim.Adam(netD.parameters(), cfg.lrD, betas=cfg.betas)
    optG = optim.Adam(netG.parameters(), cfg.lrG, betas=cfg.betas)

    if cfg.is_amp:
        [netG, netD], [optG, optD] = amp.initialize([netG, netD],
                                                    [optG, optD],
                                                    opt_level=cfg.opt_level)

    print(netG)
    print(netD)

    return netG, netD, optG, optD


def main(cfg):
    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root=cfg.dataset_dir, name=cfg.dataset_name)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # Define models and optimizers
    netG, netD, optG, optD = build_models(cfg, device)

    #########################
    #        Training
    #########################
    # Start training
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=cfg.n_dis,
        num_steps=cfg.num_steps,
        dataloader=dataloader,
        log_dir=cfg.log_dir,
        device=device)
    trainer.train()

    ##########################
    #       Evaluation
    ##########################
    # Evaluate fid
    mmc.metrics.evaluate(
        metric=cfg.metric,
        log_dir=cfg.log_dir,
        netG=netG,
        dataset_name=cfg.dataset_name,
        num_real_samples=cfg.num_real_samples,
        num_fake_samples=cfg.num_fake_samples,
        evaluate_step=cfg.evaluate_step,
        device=device)


if __name__ == "__main__":
    cfg = Config()
    main(cfg)

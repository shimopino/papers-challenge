import os
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.optim as optim
import torch_mimicry as mmc
from apex import amp
from models.fqgan_32 import FQGANGenerator, FQGANDiscriminator
from models.fqgan_pd_32 import FQGANProjectionGenerator, FQGANProjectionDiscriminator
from mimicry_wrappers import MlflowLogger


def get_models(cfg):
    if cfg.model.GAN.type == "conditional":
        netG = FQGANProjectionGenerator(num_classes=cfg.model.GAN.num_classes,
                                        nz=cfg.model.GAN.nz,
                                        ngf=cfg.model.GAN.ngf,
                                        loss_type=cfg.model.GAN.loss_type,
                                        fq_strength=cfg.model.FQ.fq_strength,
                                        is_amp=cfg.training.is_amp)
        netD = FQGANProjectionDiscriminator(num_classes=cfg.model.GAN.num_classes,
                                            ndf=cfg.model.GAN.ndf,
                                            loss_type=cfg.model.GAN.loss_type,
                                            fq_type=cfg.model.FQ.fq_type,
                                            dict_size=cfg.model.FQ.dict_size,
                                            quant_layers=cfg.model.FQ.quant_layers,
                                            fq_strength=cfg.model.FQ.fq_strength,
                                            is_amp=cfg.training.is_amp)

    else:
        netG = FQGANGenerator(nz=cfg.model.GAN.nz,
                              ngf=cfg.model.GAN.ngf,
                              loss_type=cfg.model.GAN.loss_type,
                              fq_strength=cfg.model.FQ.fq_strength,
                              is_amp=cfg.training.is_amp)
        netD = FQGANDiscriminator(ndf=cfg.model.GAN.ndf,
                                  loss_type=cfg.model.GAN.loss_type,
                                  fq_type=cfg.model.FQ.fq_type,
                                  dict_size=cfg.model.FQ.dict_size,
                                  quant_layers=cfg.model.FQ.quant_layers,
                                  fq_strength=cfg.model.FQ.fq_strength,
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


@hydra.main(config_path="./config/config.yaml")
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

    # Define mlflow logger
    # take the Child Folder name defined on config
    logger = MlflowLogger(cfg.logging.experiment_name)
    logger.log_params_from_omegaconf_dict(cfg)

    #########################
    #        Training
    #########################
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

    ##########################
    #       Evaluation
    ##########################
    # Evaluate fid
    fid_metrics = mmc.metrics.evaluate(
        metric="fid",
        log_dir=cfg.logging.log_dir,
        netG=netG,
        dataset_name=cfg.dataset.dataset_name,
        num_real_samples=cfg.metrics.fid.num_real_samples,
        num_fake_samples=cfg.metrics.fid.num_fake_samples,
        evaluate_step=cfg.metrics.fid.evaluate_step,
        device=device)

    fid_score = np.mean(fid_metrics[cfg.metrics.fid.evaluate_step])
    logger.log_metric("fid", fid_score)

    # Evaluate inception score
    inception_score_metrics = mmc.metrics.evaluate(
        metric='inception_score',
        log_dir=cfg.logging.log_dir,
        netG=netG,
        num_samples=cfg.metrics.inception_score.num_samples,
        evaluate_step=cfg.metrics.inception_score.evaluate_step,
        device=device)

    inception_score_score = np.mean(inception_score_metrics[cfg.metrics.inception_score.evaluate_step])
    logger.log_metric("inception score", inception_score_score)

    # Evaluate kid
    kid_metrics = mmc.metrics.evaluate(
        metric='kid',
        log_dir=cfg.logging.log_dir,
        netG=netG,
        dataset_name=cfg.dataset.dataset_name,
        num_subsets=cfg.metrics.kid.num_subsets,
        subset_size=cfg.metrics.kid.subset_size,
        evaluate_step=cfg.metrics.kid.evaluate_step,
        device=device)

    kid_score = np.mean(kid_metrics[cfg.metrics.kid.evaluate_step])
    logger.log_metric("kid", kid_score)

    # add evaluation metrics to mlflow
    logger.log_torch_model(netG.cpu(), "netG")
    logger.log_torch_model(netD.cpu(), "netD")

    # Add Hydra output to Artifact
    logger.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    logger.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    logger.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    logger.log_artifact(os.path.join(os.getcwd(), 'main.log'))
    logger.set_terminated()


if __name__ == "__main__":
    main()

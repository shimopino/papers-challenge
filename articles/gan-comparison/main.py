import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torch_mimicry as mmc
from models.dcgan import DCGANGenerator, DCGANDiscriminator


if __name__ == "__main__":

    log_dir = "./logs"
    seed = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transforms = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    custom_dataset = ImageFolder(
        root="input/animeface-character-dataset/thumb/",
        transform=transforms
    )

    dataloader = DataLoader(
        dataset=custom_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True
    )

    netG = DCGANGenerator().to(device)
    netD = DCGANDiscriminator().to(device)
    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

    # Start training
    trainer = mmc.training.Trainer(netD=netD,
                                   netG=netG,
                                   optD=optD,
                                   optG=optG,
                                   n_dis=5,
                                   num_steps=100000,
                                   lr_decay='linear',
                                   dataloader=dataloader,
                                   log_dir=log_dir,
                                   device=device)
    trainer.train()

    # Metrics with a custom dataset.
    mmc.metrics.fid_score(num_real_samples=10000,
                          num_fake_samples=10000,
                          netG=netG,
                          seed=seed,
                          dataset=custom_dataset,
                          log_dir=log_dir,
                          device=device,
                          stats_file=log_dir + '/fid_stats.npz')

    mmc.metrics.kid_score(num_samples=10000,
                          netG=netG,
                          seed=seed,
                          dataset=custom_dataset,
                          log_dir=log_dir,
                          device=device,
                          feat_file=log_dir + '/kid_stats.npz')

    mmc.metrics.evaluate(metric='fid',
                        log_dir='./log/sngan_example/',
                        netG=netG,
                        dataset=custom_dataset,
                        num_real_samples=10000,
                        num_fake_samples=10000,
                        evaluate_step=100000,
                        stats_file=log_dir + '/fid_stats.npz',
                        device=device)


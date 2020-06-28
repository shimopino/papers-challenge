import torch
import torch.nn as nn

from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock
from torch_mimicry.nets.sngan import sngan_base

from .resblocks import SGBlock
from .layers import SBN


class SBNSNGANGenerator32(sngan_base.SNGANGenerator):

    def __init__(self, nz=128, ngf=256, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        # build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = SGBlock(self.ngf, self.ngf, upsample=True, nz=self.nz)
        self.block3 = SGBlock(self.ngf, self.ngf, upsample=True, nz=self.nz)
        self.block4 = SGBlock(self.ngf, self.ngf, upsample=True, nz=self.nz)
        self.b5 = SBN(self.ngf, self.nz)
        self.c5 = nn.Conv2d(self.ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.

        Args:
            x (torch.Tensor): A batch of noise vectors of shape (N, nz).

        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """

        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, x)
        h = self.block3(h, x)
        h = self.block4(h, x)
        h = self.b5(h, x)
        h = self.activation(h)
        h = torch.tanh(self.c5(h))

        return h

    def generate_images(self, num_images, device=None):
        r"""
        Generates num_images randomly.

        Args:
            num_images (int): Number of images to generate
            device (torch.device): Device to send images to.

        Returns:
            Tensor: A batch of generated images.
        """

        if device is None:
            device = self.device

        noise = torch.randn((num_images, self.nz), device=device)
        fake_images = self.forward(noise)

        return fake_images

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.
        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (dict): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.
        Returns:
            Returns MetricLog object containing updated logging variables after 1 training step.
        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake images
        fake_images = self.generate_images(num_images=batch_size,
                                           device=device)

        # Compute output logit of D thinking image real
        output = netD(fake_images)

        # Compute loss
        errG = self.compute_gan_loss(output=output)

        # Backprop and update gradients
        errG.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data


class SBNSNGANDiscriminator32(sngan_base.SNGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf=128, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf)
        self.block2 = DBlock(self.ndf, self.ndf, downsample=True)
        self.block3 = DBlock(self.ndf, self.ndf, downsample=False)
        self.block4 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l5 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)

        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)

        return output

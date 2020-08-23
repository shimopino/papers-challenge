import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from torch_mimicry.nets import gan
from torch_mimicry.modules import GBlock, DBlock, DBlockOptimized
from torch_mimicry.modules import SNLinear, SNConv2d

from .modules import SelfAttention, DBlockDecoder


class UNetGenerator(gan.BaseGenerator):
    r"""
    ResNet backbone generator for SNGAN.
    You can select the non-saturating loss only.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """

    def __init__(self, nz=128, ngf=256, bottom_width=4, loss_type="ns", is_amp=False, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, loss_type=loss_type, **kwargs)

        assert self.loss_type == "ns", "yet, you can select [ns] only"

        self.is_amp = is_amp

        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.activation = nn.ReLU(inplace=True)
        self.c5 = nn.Conv2d(self.ngf, 3, kernel_size=3, stride=1, padding=1)

        # initialize the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

        # self-attention
        self.non_local_block = SelfAttention(self.ngf, spectral_norm=False)

    def forward(self, x):
        r"""
        feedforward a batch of noise vectors into a batch of fake images.

        Args:
            x (torch.Tensor): a batch of noise vector of shape (N, nz)

        Returns:
            output (torch.Tensor): a batch of fake images of shape (N, C, H, W)
        """

        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.non_local_block(h)
        h = self.block4(h)
        h = self.activation(self.b5(h))
        h = torch.tanh(self.c5(h))

        return h

    def compute_seg_loss(self, output):
        r"""
        Computes Segmentation loss for generator.

        Args:
            output (Tensor): A batch of output segmentation logits
                from the discriminator of shape (N, 1, 32, 32).

        Returns:
            Tensor: A batch of Segmentation losses for the generator.
        """

        # calculate all pixel prediction of [0(fake), 1(real)]
        output_fake = torch.sigmoid(output)
        # calculate mean of all pixels (N, 1, 32, 32) --> (N, 1)
        output_fake = torch.log(output_fake + 1e-8).mean()

        loss = -torch.mean(output_fake)

        return loss

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

        ################################
        # Apply UNet-based Discriminator
        ################################
        # Compute output logit of D thinking image real
        # output, output_seg = netD(fake_images)
        output, _ = netD(fake_images)

        # Compute GAN loss
        errG = self.compute_gan_loss(output=output)
        # add Segmentation loss
        # errG = errG + self.compute_seg_loss(output=output_seg)

        # backprop
        if self.is_amp:
            with torch.autograd.set_detect_anomaly(True):
                with amp.scale_loss(errG, optG) as scaled_loss:
                    scaled_loss.backward()
        else:
            with torch.autograd.set_detect_anomaly(True):
                errG.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data


class UNetDiscriminator(gan.BaseDiscriminator):
    r"""
    ResNet backbone Unet-based discriminator.
    You can select the non-saturating loss only.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
        cutmix (boolean): If True, use CutMix Augmentation.
        consistency (boolean): If True, use Consistency Reglarization.
        warmup (int): the warmup iteration count for linearly increasing the cutmix probability.
    """

    def __init__(self, ndf=128, loss_type="ns", is_amp=False, cutmix=False, consistency=False, warmup=50000, **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
        self.is_amp = is_amp
        self.cutmix = cutmix
        self.consistency = consistency

        # build encoder
        self.down1 = DBlockOptimized(3, self.ndf)
        self.down2 = DBlock(self.ndf, self.ndf, downsample=True)
        self.down3 = DBlock(self.ndf, self.ndf, downsample=False)
        self.down4 = DBlock(self.ndf, self.ndf, downsample=False)
        self.down_act = nn.ReLU(inplace=True)
        self.down_l5 = SNLinear(self.ndf, 1)
        # self-attention
        self.non_local_block = SelfAttention(self.ndf, spectral_norm=True)
        # initialize weights
        nn.init.xavier_uniform_(self.down_l5.weight.data, 1.0)

        # build decoder
        self.up1 = DBlockDecoder(self.ndf, self.ndf, upsample=False)
        self.up2 = DBlockDecoder(self.ndf * 2, self.ndf, upsample=False)
        self.up3 = DBlockDecoder(self.ndf * 2, self.ndf, upsample=True)
        self.up4 = DBlockDecoder(self.ndf * 2, self.ndf, upsample=True)
        self.up_act = nn.ReLU(inplace=True)
        self.up_c5 = SNConv2d(self.ndf, 1, kernel_size=3, stride=1, padding=1)
        # initialize weights
        nn.init.xavier_uniform_(self.up_c5.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            dis_logits (Tensor): A batch of GAN logits of shape (N, 1).
            seg_logits (Tensor): A batch of Segmentation logits of shape (B, 1, 32, 32).
        """

        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h3 = self.non_local_block(h3)
        h4 = self.down4(h3)
        # h = self.up1(h4)
        # h = self.up2(torch.cat((h, h3), dim=1))
        # h = self.up3(torch.cat((h, h2), dim=1))
        # h = self.up4(torch.cat((h, h1), dim=1))

        # discriminative logits
        dis_out = torch.sum(h4, dim=(2, 3))
        dis_logits = self.down_l5(dis_out)

        # segmentation logits
        # seg_out = self.up_act(h)
        # seg_logits = self.up_c5(seg_out)

        # return dis_logits, seg_logits
        return dis_logits, None


    def compute_seg_loss(self, output_real, output_fake, real_label_val=1.0, fake_label_val=0.0):
        r"""
        Computes Segmentation loss for discriminator.

        Args:
            output_real (Tensor): A batch of output segmentation logits of shape (N, 1, 32, 32) from real images.
            output_fake (Tensor): A batch of output segmentation logits of shape (N, 1, 32, 32) from fake images.
            real_label_val (int): Label for real images.
            fake_label_val (int): Label for fake images.

        Returns:
            errD (Tensor): A batch of GAN losses for the discriminator.
        """

        # produce real and fake labels
        fake_labels = torch.full((output_fake.shape[0], 1, 32, 32),
                                 fake_label_val,
                                 device=output_fake.device)
        real_labels = torch.full((output_real.shape[0], 1, 32, 32),
                                 real_label_val,
                                 device=output_real.device)

        # compute loss
        errD_fake = F.binary_cross_entropy_with_logits(output_fake, fake_labels)
        errD_real = F.binary_cross_entropy_with_logits(output_real, real_labels)

        loss = errD_real + errD_fake

        return loss

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""__init__
        Takes one training step for D.
        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
            loss_type (str): Name of loss to use for GAN loss.
            netG (nn.Module): Generator model for obtaining fake images.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (dict): A dict mapping name to values for logging uses.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.
        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.
        """

        self.zero_grad()
        real_images, real_labels = real_batch
        batch_size = real_images.shape[0]  # Match batch sizes for last iter

        # Produce fake images
        fake_images = netG.generate_images(num_images=batch_size,
                                           device=device).detach()

        ################################
        # Apply UNet-based Discriminator
        ################################
        # Produce logits for real images
        # output_real, output_real_seg = self.forward(real_images)
        output_real, _ = self.forward(real_images)

        # Produce logits for fake images
        output_fake, output_fake_seg = self.forward(fake_images)
        output_fake, _ = self.forward(fake_images)

        # Compute loss for D
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)
        # compute Segmentation loss
        # errD = errD + self.compute_seg_loss(output_real=output_real_seg,
        #                                     output_fake=output_fake_seg)

        # Backprop and update gradients
        if self.is_amp:
            with torch.autograd.set_detect_anomaly(True):
                with amp.scale_loss(errD, optD) as scaled_loss:
                    scaled_loss.backward()
        else:
            with torch.autograd.set_detect_anomaly(True):
                errD.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # Log statistics for D once out of loop
        log_data.add_metric('errD', errD.item(), group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data


if __name__ == "__main__":
    inputs = torch.randn(10, 3, 32, 32)

    # Discriminator forward
    netD = UNetDiscriminator(ndf=128, loss_type="ns", is_amp=True)
    output, output_seg = netD(inputs)
    print(output.shape)
    # print(output_seg.shape)

    # Discriminator backward
    from torch.nn import functional as F
    dis_loss = F.mse_loss(output, torch.ones((10, 1)))
    seg_loss = F.mse_loss(output_seg, torch.ones((10, 1, 32, 32)))
    loss = dis_loss + seg_loss
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()

    # Generator forward
    noise = torch.randn(10, 128)
    netG = UNetGenerator(nz=128, ngf=256, bottom_width=4, loss_type="ns", is_amp=True)
    output = netG(noise)
    print(output.shape)

    # Generator backward
    loss = F.mse_loss(output, torch.ones((10, 1, 32, 32)))
    loss.backward()

    from torch.optim import Adam
    optG = Adam(netG.parameters(), lr=0.01)
    optD = Adam(netD.parameters(), lr=0.01)

    from torch_mimicry.training import MetricLog
    log_data = MetricLog()

    netG.train_step((inputs, ), netD, optG, log_data)

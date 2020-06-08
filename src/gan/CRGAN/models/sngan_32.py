import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from torch_mimicry.nets import gan
from torch_mimicry.modules import SNLinear, GBlock, DBlock, DBlockOptimized

from modules import RandomResizedCropTensor, RandomHorizontalFlipTensor


class ICRGenerator(gan.BaseGenerator):

    def __init__(self,
                 nz=128,
                 ngf=256,
                 bottom_width=4,
                 loss_type="hinge",
                 is_amp=False,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)

        self.is_amp = is_amp

        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.activation = nn.ReLU(inplace=True)
        self.c5 = nn.Conv2d(self.ngf, 3, kernel_size=3, stride=1, padding=1)

        # initialize the weights
        nn.init.xavier_normal_(self.l1.weight.data, 1.0)
        nn.init.xavier_normal_(self.c5.weight.data, 1.0)

    def forward(self, x):

        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(self.b5(h))
        h = self.c5(h)
        output = torch.tanh(h)

        return output

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):

        self.zero_grad()

        batch_size = real_batch[0].shape(0)

        fake_images = self.generate_images(num_images=batch_size,
                                           device=device)

        output = netD(fake_images)

        errG = self.compute_gan_loss(output)

        # backprop
        if self.is_amp:
            with amp.scale_loss(errG, optG) as scaled_loss:
                scaled_loss.backward()
        else:
            errG.backward()

        # update weights
        optG.step()

        # log_data
        log_data.add_metric("errG", errG, group="loss")

        return log_data


class ICRDiscriminator(gan.BaseDiscriminator):

    def __init__(self,
                 ndf=128,
                 loss_type="hinge",
                 is_amp=False,
                 is_transforms=False,
                 real_lambda=10,
                 fake_lambda=5,
                 **kwargs):
        super().__init__(ndf=ndf,
                         loss_type=loss_type,
                         **kwargs)

        self.is_amp = is_amp
        self.is_transforms = is_transforms
        self.real_lambda = real_lambda
        self.fake_lambda = fake_lambda

        self.block1 = DBlockOptimized(3, self.ndf)
        self.block2 = DBlock(self.ndf, self.ndf, downsample=True)
        self.block3 = DBlock(self.ndf, self.ndf, downsample=False)
        self.block4 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l5 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(inplace=True)

        # initialize the weights
        nn.init.xavier_normal_(self.l5.weight.data, 1.0)

    def forward(self, x):

        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)

        # global average pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)

        return output

    def apply_transforms(self, tensor):
        """
        apply RandomResizedCrop and RandomHorizontalFlip to 4D-Tensorflow

        Parameters
        ----------
        tensor : torch.Tensor
            input image tensor like (B, C, H, W)

        Returns
        -------
        torch.Tensor
            image tensor that RandomResizedCrop and RandomHorizontalFlip apply
        """

        H, W = tensor.shape[2], tensor.shape[3]

        tensor = RandomResizedCropTensor(size=(H, W))(tensor)
        tensor = RandomHorizontalFlipTensor()(tensor)

        return tensor

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):

        self.zero_grad()

        #########################
        # get input image tensor
        #########################
        # get real images
        real_images, _ = real_batch
        batch_size = real_images.shape[0]

        # get fake images
        fake_images = netG.generate_images(num_images=batch_size,
                                           device=device).detach()

        ##################
        # Generative Loss
        ##################
        # compute real and fake logits
        output_real = self.forward(real_images)
        output_fake = self.forward(fake_images)

        # compute gan loss
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        if self.is_transforms:
            #########################
            # apply data augmentation
            #########################
            augmented_real_images = self.apply_transforms(real_images)
            augmented_fake_images = self.apply_transforms(fake_images)

            #####################################
            # Balanced Consistency Regularization
            #####################################
            augmented_output_real = self.forward(augmented_real_images)
            augmented_output_fake = self.forward(augmented_fake_images)

            # conpute regularization loss
            errD_augmented_real = F.mse_loss(input=output_real,
                                             target=augmented_output_real)
            errD_augmented_fake = F.mse_loss(input=output_fake,
                                             target=augmented_output_fake)

        if self.is_transforms:
            # sum all errD
            total_errD = errD + self.real_lambda * errD_augmented_real + self.fake_lambda * errD_augmented_fake
        else:
            total_errD = errD

        if self.is_amp:
            with amp.scale_loss(total_errD, optD) as scaled_loss:
                scaled_loss.backward()
        else:
            total_errD.backward()

        # update weights
        optD.step()

        # compute probs
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # log statistics
        log_data.add_metric("errD", errD, group="loss")
        log_data.add_metric("errD_augmented_real", errD_augmented_real, group="loss")
        log_data.add_metric("errD_augmented_fake", errD_augmented_fake, group="loss")
        log_data.add_metric("D(x)", D_x, group="prob")
        log_data.add_metric("D(G(x))", D_Gz, group="prob")

        return log_data


if __name__ == "__main__":

    netG = ICRGenerator()
    netD = ICRDiscriminator(is_transforms=True)

    inputG = torch.randn(100, 128)
    outputG = netG(inputG)

    print("input: ", inputG.shape)
    print("output: ", outputG.shape)

    inputD = torch.randn(100, 3, 32, 32)
    outputD = netD(inputD)

    print("input: ", inputD.shape)
    print("output: ", outputD.shape)

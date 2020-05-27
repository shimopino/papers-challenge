import torch
import torch.nn as nn
from .modules import SNLinear, SNConv2d
from .resblocks import GBlock, DBlock, DBlockOptimized
from .vq import VectorQuantizerEMA, VectorQuantizer


class ResNetGenerator(nn.Module):
    def __init__(
        self,
        nz,
        ngf,
        nc,
        bottom_width=4,
        spectral_norm=False
    ):
        """
        Generator which has Resblock for resolution 64x64

        Args:
            nz (int): The channel size of input latent code.
            ngf (int): The base channel size of each GBlock.
            nc (int): The channel size of output image.
            bottom_width (int, optional): The output image size of the bottom layer. Defaults to 4.
            spectral_norm (bool, optional): If True, uses spectral norm for convolutional layers. Defaults to False.
        """
        super(ResNetGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.bottom_width = bottom_width
        self.spectral_norm = spectral_norm

        # [B, nz] --> [B, (width*width)*ngf] --> [B, ngf, width, width]
        if self.spectral_norm:
            self.linear1 = SNLinear(self.nz, (self.bottom_width ** 2) * self.ngf)
        else:
            self.linear1 = nn.Linear(self.nz, (self.bottom_width ** 2) * self.ngf)
        # [B, ngf, width, width] --> [B, ngf, width*2, width*2]
        self.block2 = GBlock(self.ngf >> 0, self.ngf >> 1, upsample=True, spectral_norm=spectral_norm)
        # [B, ngf, width*2, width*2] --> [B, ngf, width*4, width*4]
        self.block3 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True, spectral_norm=spectral_norm)
        # [B, ngf, width*4, width*4] --> [B, ngf, width*8, width*8]
        self.block4 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True, spectral_norm=spectral_norm)
        # [B, ngf, width*8, width*8] --> [B, ngf, width*16, width*16]
        self.block5 = GBlock(self.ngf >> 3, self.ngf >> 4, upsample=True, spectral_norm=spectral_norm)
        self.bn6 = nn.BatchNorm2d(self.ngf >> 4)
        # set last conv for logits
        if self.spectral_norm:
            self.conv6 = SNConv2d(self.ngf >> 4, self.nc, kernel_size=3, stride=1, padding=1)
        else:
            self.conv6 = nn.Conv2d(self.ngf >> 4, self.nc, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU(inplace=True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.conv6.weight.data, 1.0)

    def forward(self, x):

        h = self.linear1(x)
        # [B, ngf, 4,  4]
        h = h.view(-1, self.ngf, self.bottom_width, self.bottom_width).contiguous()
        h = self.block2(h)  # [B, ngf//2,   8,  8]
        h = self.block3(h)  # [B, ngf//4,  16, 16]
        h = self.block4(h)  # [B, ngf//8,  32, 32]
        h = self.block5(h)  # [B, ngf//16, 64, 64]
        h = self.bn6(h)
        h = self.activation(h)
        h = torch.tanh(self.conv6(h))  # [B, 3, 64, 64]

        return h

    def sample_latent(self, batch_size):
        """
        Return latent code of the dimension [batch_size, self.nz].
        """
        return torch.randn((batch_size, self.nz))

    def generate_images(self, num_images, device=None):
        """
        Return Generated images of the arbitary number. This requires the device setting.
        """
        assert device is not None, "please set device setting"

        noise = self.sample_latent((num_images)).to(device)
        fake_images = self.forward(noise)

        return fake_images


class ResNetDiscriminator(nn.Module):
    def __init__(
        self,
        nc,
        ndf,
        spectral_norm=False,
        vq_type=None,
        dict_size=5,
        quant_layers=None,
    ):
        """
        Discriminator which has Resblock for resolution 64x64 with  Feature Quantization module.

        Args:
            nc (int): The channel size of input image
            ndf ([type]): The base channel size of each DBlock
            spectral_norm (bool, optional): If True, uses spectral norm for convolutional layers. Defaults to False.
            vq_type (str, optional): The VQ module type ["Normal", "EMA"]. Defaults to None.
            dict_size (int, optional): The number of dictionary vector. Defaults to 5.
            quant_layers (list, optional): The Layer that VQ module apply . Defaults to None.
        """

        super(ResNetDiscriminator, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.spectral_norm = spectral_norm
        self.vq_type = vq_type
        self.dict_size = dict_size
        self.quant_layers = quant_layers

        if self.vq_type is not None:
            assert self.vq_type in ['Normal', 'EMA'], "set vq_type within ['Normal', 'EMA']"

        self.block1 = DBlockOptimized(self.nc, self.ndf >> 4, spectral_norm=self.spectral_norm)
        self.block2 = DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True, spectral_norm=self.spectral_norm)
        self.block3 = DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True, spectral_norm=self.spectral_norm)
        self.block4 = DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True, spectral_norm=self.spectral_norm)
        self.block5 = DBlock(self.ndf >> 1, self.ndf >> 0, downsample=True, spectral_norm=self.spectral_norm)

        if self.spectral_norm:
            self.linear6 = SNLinear(self.ndf, 1)
        else:
            self.linear6 = nn.Linear(self.ndf, 1)
        self.activation = nn.ReLU(inplace=True)

        if self.vq_type == "Normal":
            self.vq = VectorQuantizer(self.ndf >> 2, 2 ** self.dict_size)
        elif self.vq_type == "EMA":
            self.vq = VectorQuantizerEMA(self.ndf >> 2, 2 ** self.dict_size)

        # if self.vq_type:
        #     assert quant_layers is not None, "should set quant_layers like ['3']"
        #     assert (min(quant_layers) > 1) and (max(quant_layers) < 6), "should be range [2, 5]"
        #     for layer in quant_layers:
        #         out_channels = getattr(self, f"block{layer}").out_channels
        #         if self.vq_type == "Normal":
        #             setattr(self, f"vq{layer}", VectorQuantizer(out_channels, 2 ** self.dict_size))
        #         elif self.vq_type == "EMA":
        #             setattr(self, f"vq{layer}", VectorQuantizerEMA(out_channels, 2 ** self.dict_size))

        # Initialise the weights
        nn.init.xavier_uniform_(self.linear6.weight.data, 1.0)

    def forward(self, x):

        h = x
        h = self.block1(h)

        # quant_loss = 0
        # for layer in range(2, 6):
        #     h = getattr(self, f"block{layer}")(h)
        #     # apply Feature Quantization
        #     if (self.vq_type) and (layer in self.quant_layers):
        #         h, loss, embed_idx = getattr(self, f"vq{layer}")(h)
        #         quant_loss += loss

        h = self.block2(h)
        h = self.block3(h)
        if self.vq_type:
            h, loss, embed_idx = self.vq(h)
        h = self.block4(h)
        h = self.block5(h)

        h = self.activation(h)
        # Global average pooling
        h = torch.sum(h, dim=(2, 3))  # [B, ndf]
        output = self.linear6(h).squeeze()

        if self.vq_type:
            return output, loss, embed_idx
        else:
            return output

    def compute_probs(self, output):
        """
        Compute Probability of the input Logits
        """

        prob_x = torch.sigmoid(output).mean().item()

        return prob_x


if __name__ == "__main__":
    netG = ResNetGenerator(100, 1024, 3, 4, True)
    print(netG(torch.randn(10, 100)))

    netD = ResNetDiscriminator(3, 1024, True, True, 5)
    print(netD(torch.randn(10, 3, 64, 64)))

    sample = getattr(netD, "block2")
    print(sample)
    print(sample.in_channels)
    print(sample.out_channels)

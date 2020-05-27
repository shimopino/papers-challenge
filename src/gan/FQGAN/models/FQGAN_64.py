import torch
import torch.nn as nn
from .modules import SNConv2d, SNConvTranspose2d
from .vq import VectorQuantizerEMA, VectorQuantizer


class Generator(nn.Module):
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
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.bottom_width = bottom_width
        self.spectral_norm = spectral_norm

        # [B, nz] --> [B, (width*width)*ngf] --> [B, ngf, width, width]
        if self.spectral_norm:
            self.layer1 = nn.Sequential(
                SNConvTranspose2d(self.nz, self.ngf >> 0, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.ngf >> 0),
                nn.ReLU(inplace=False)
            )
            self.layer2 = nn.Sequential(
                SNConvTranspose2d(self.ngf >> 0, self.ngf >> 1, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf >> 1),
                nn.ReLU(inplace=False)
            )
            self.layer3 = nn.Sequential(
                SNConvTranspose2d(self.ngf >> 1, self.ngf >> 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf >> 2),
                nn.ReLU(inplace=False)
            )
            self.layer4 = nn.Sequential(
                SNConvTranspose2d(self.ngf >> 2, self.ngf >> 3, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf >> 3),
                nn.ReLU(inplace=False)
            )
            self.layer5 = nn.Sequential(
                SNConvTranspose2d(self.ngf >> 3, self.ngf >> 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf >> 4),
                nn.ReLU(inplace=False)
            )
            self.layer6 = nn.Sequential(
                SNConv2d(self.ngf >> 4, self.nc, 3, 1, 1, bias=False),
                nn.Tanh()
            )

        else:
            self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(self.nz, self.ngf >> 0, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.ngf >> 0),
                nn.ReLU(inplace=False)
            )
            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(self.ngf >> 0, self.ngf >> 1, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf >> 1),
                nn.ReLU(inplace=False)
            )
            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(self.ngf >> 1, self.ngf >> 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf >> 2),
                nn.ReLU(inplace=False)
            )
            self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(self.ngf >> 2, self.ngf >> 3, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf >> 3),
                nn.ReLU(inplace=False)
            )
            self.layer5 = nn.Sequential(
                nn.ConvTranspose2d(self.ngf >> 3, self.ngf >> 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf >> 4),
                nn.ReLU(inplace=False)
            )
            self.layer6 = nn.Sequential(
                nn.Conv2d(self.ngf >> 4, self.nc, 3, 1, 1, bias=False),
                nn.Tanh()
            )

    def forward(self, x):

        h = x.view(-1, self.nz, 1, 1)
        h = self.layer1(h)  # [B, ngf,      4,  4]
        h = self.layer2(h)  # [B, ngf//2,   8,  8]
        h = self.layer3(h)  # [B, ngf//4,  16, 16]
        h = self.layer4(h)  # [B, ngf//8,  32, 32]
        h = self.layer5(h)  # [B, ngf//16, 64, 64]
        h = self.layer6(h)

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


class Discriminator(nn.Module):
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

        super().__init__()

        self.nc = nc
        self.ndf = ndf
        self.spectral_norm = spectral_norm
        self.vq_type = vq_type
        self.dict_size = dict_size
        self.quant_layers = quant_layers

        if self.vq_type is not None:
            assert self.vq_type in ['Normal', 'EMA'], "set vq_type within ['Normal', 'EMA']"

        if self.spectral_norm:
            self.layer1 = nn.Sequential(
                SNConv2d(self.nc, self.ndf >> 4, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.layer2 = nn.Sequential(
                SNConv2d(self.ndf >> 4, self.ndf >> 3, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf >> 3),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.layer3 = nn.Sequential(
                SNConv2d(self.ndf >> 3, self.ndf >> 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf >> 2),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.layer4 = nn.Sequential(
                SNConv2d(self.ndf >> 2, self.ndf >> 1, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf >> 1),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.layer5 = nn.Sequential(
                SNConv2d(self.ndf >> 1, self.ndf >> 0, 4, 2, 0, bias=False),
                nn.BatchNorm2d(self.ndf >> 0),
            )
            self.layer6 = nn.Sequential(
                SNConv2d(self.ndf >> 0, 1, 3, 1, 1, bias=False),
            )

        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(self.nc, self.ndf >> 4, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(self.ndf >> 4, self.ndf >> 3, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf >> 3),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(self.ndf >> 3, self.ndf >> 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf >> 2),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(self.ndf >> 2, self.ndf >> 1, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf >> 1),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(self.ndf >> 1, self.ndf >> 0, 4, 2, 0, bias=False),
                nn.BatchNorm2d(self.ndf >> 0),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.layer6 = nn.Sequential(
                nn.Conv2d(self.ndf >> 0, 1, 3, 1, 1, bias=False),
            )

        if self.vq_type == "Normal":
            self.vq = VectorQuantizer(self.ndf >> 2, 2 ** self.dict_size)
        elif self.vq_type == "EMA":
            self.vq = VectorQuantizerEMA(self.ndf >> 2, 2 ** self.dict_size)

    def forward(self, x):

        h = x
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        if self.vq_type:
            h, loss, embed_idx = self.vq(h)
        h = self.layer4(h)
        h = self.layer5(h)
        output = self.layer6(h).view(-1)

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
    netG = Generator(100, 1024, 3, 4, True)
    print(netG)
    print(netG(torch.randn(10, 100)).shape)

    netD = Discriminator(3, 1024, True)
    print(netD)
    print(netD(torch.randn(10, 3, 64, 64)).shape)

    netD = Discriminator(3, 1024, True, "Normal", dict_size=10)
    print(netD)
    print(netD(torch.randn(10, 3, 64, 64))[0].shape)

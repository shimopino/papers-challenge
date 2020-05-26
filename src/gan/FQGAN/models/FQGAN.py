import torch
import torch.nn as nn
from .modules import SNLinear, SNConv2d
from .resblocks import GBlock, DBlock, DBlockOptimized
from .vq import VectorQuantizerEMA


class ResNetGenerator(nn.Module):
    def __init__(self, nz, ngf, nc, bottom_width=4, use_sn=True):
        super(ResNetGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.bottom_width = bottom_width

        # [B, nz] --> [B, (width*width)*ngf] --> [B, ngf, width, width]
        self.l1 = SNLinear(use_sn, self.nz, (self.bottom_width ** 2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf >> 1, upsample=True, use_sn=use_sn)
        self.block3 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True, use_sn=use_sn)
        self.block4 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True, use_sn=use_sn)
        self.block5 = GBlock(self.ngf >> 3, self.ngf >> 4, upsample=True, use_sn=use_sn)
        self.b6 = nn.BatchNorm2d(self.ngf >> 4)
        self.c6 = SNConv2d(
            use_sn, self.ngf >> 4, self.nc, kernel_size=3, stride=1, padding=1
        )
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c6.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        # [B, ngf, 4,  4]
        h = h.view(x.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)  # [B, ngf//2,   8,  8]
        h = self.block3(h)  # [B, ngf//4,  16, 16]
        h = self.block4(h)  # [B, ngf//8,  32, 32]
        h = self.block5(h)  # [B, ngf//16, 64, 64]
        h = self.b6(h)
        h = self.activation(h)
        h = torch.tanh(self.c6(h))  # [B, 3, 64, 64]

        return h

    def sample_latent(self, batch_size):
        return torch.randn((batch_size, self.nz))

    def generate_images(self, num_images, device=None):
        assert device is not None, "please set device setting"

        noise = self.sample_latent((num_images)).to(device)
        fake_images = self.forward(noise)

        return fake_images


class ResNetDiscriminator(nn.Module):
    def __init__(
        self, nc, ndf, use_sn=False, use_vq=True, dict_size=5, quant_layers=None
    ):
        super(ResNetDiscriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.use_vq = use_vq
        self.dict_size = dict_size
        self.quant_layers = quant_layers

        self.block1 = DBlockOptimized(self.nc, self.ndf >> 4, use_sn=use_sn)
        self.block2 = DBlock(
            self.ndf >> 4, self.ndf >> 3, downsample=True, use_sn=use_sn
        )
        self.block3 = DBlock(
            self.ndf >> 3, self.ndf >> 2, downsample=True, use_sn=use_sn
        )
        self.block4 = DBlock(
            self.ndf >> 2, self.ndf >> 1, downsample=True, use_sn=use_sn
        )
        self.block5 = DBlock(self.ndf >> 1, self.ndf, downsample=True, use_sn=use_sn)
        self.l6 = SNLinear(use_sn, self.ndf, 1)
        self.activation = nn.ReLU(True)

        if self.use_vq:
            assert quant_layers is not None, "should set quant_layers like ['3']"
            assert (min(quant_layers) > 1) and (
                max(quant_layers) < 6
            ), "should be range [2, 5]"
            for layer in quant_layers:
                out_channels = getattr(self, f"block{layer}").out_channels
                setattr(
                    self,
                    f"vq{layer}",
                    VectorQuantizerEMA(out_channels, 2 ** self.dict_size),
                )

        # Initialise the weights
        nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

    def forward(self, x):

        h = x
        h = self.block1(h)  # [B, ndf//16, 64, 64]

        # h = self.block2(h)  # [B, ndf//8,  32, 32]
        # h = self.block3(h)  # [B, ndf//4,  16, 16]
        # if self.use_vq:
        #     h, loss, ppl = self.vq(h)
        # h = self.block4(h)  # [B, ndf//2,   8,  8]
        # h = self.block5(h)  # [B, ndf,      4,  4]

        quant_loss = 0
        for layer in range(2, 6):
            h = getattr(self, f"block{layer}")(h)

            if (self.use_vq) and (layer in self.quant_layers):
                h, loss, ppl, embed_idx = getattr(self, f"vq{layer}")(h)
                quant_loss += loss

        h = self.activation(h)
        # Global average pooling
        h = torch.sum(h, dim=(2, 3))  # [B, ndf]
        output = self.l6(h).squeeze()

        if self.use_vq:
            return output, loss, ppl, embed_idx
        else:
            return output, None, None, None

    def compute_probs(self, output_real, output_fake):
        D_x = torch.sigmoid(output_real).mean().item()
        D_Gz = torch.sigmoid(output_fake).mean().item()

        return D_x, D_Gz


if __name__ == "__main__":
    netG = ResNetGenerator(100, 1024, 3, 4, True)
    print(netG)

    netD = ResNetDiscriminator(3, 1024, True, True, 5)
    print(netD)

    sample = getattr(netD, "block2")
    print(sample)
    print(sample.in_channels)
    print(sample.out_channels)

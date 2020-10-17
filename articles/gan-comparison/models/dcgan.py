import torch.nn as nn
from torch_mimicry.nets.gan import gan


class DCGANGenerator(gan.BaseGenerator):

    def __init__(self, 
        nz=128, 
        ngf=1024, 
        bottom_width=4, 
        loss_type="ns",
        **kwargs
    ):

        super().__init__(
            nz=nz, 
            ngf=ngf, 
            bottom_width=bottom_width, 
            loss_type=loss_type, 
            **kwargs
        )

        # [B, nz, 1, 1] --> [B, ngf, 4, 4]
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU()
        )

        # [B, ngf, 4, 4] --> [B, ngf/2, 8, 8]
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf >> 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf >> 1),
            nn.ReLU()
        )

        # [B, ngf/2, 8, 8] --> [B, ngf/4, 16, 16]
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf >> 1, ngf >> 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf >> 2),
            nn.ReLU()
        )

        # [B, ngf/4, 16, 16] --> [B, ngf/8, 32, 32]
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf >> 2, ngf >> 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf >> 3),
            nn.ReLU()
        )

        # [B, ngf/8, 32, 32] --> [B, ngf/16, 64, 64]
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf >> 3, ngf >> 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf >> 4),
            nn.ReLU()
        )

        # [B, ngf/16, 64, 64] --> [B, 3, 128, 128]
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(ngf >> 4, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, y=None):

        x = x.view(-1, self.nz, 1, 1)
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.layer6(h)

        return h


class DCGANDiscriminator(gan.BaseDiscriminator):

    def __init__(self, 
        nc=3,
        ndf=1024, 
        loss_type="ns", 
        **kwargs
    ):

        super().__init__(
            ndf=ndf, 
            loss_type=loss_type, 
            **kwargs
        )

        # [B, 3, 128, 128] --> [B, 64, 64, 64]
        self.layer1 = nn.Sequential(
            nn.Conv2d(nc, ndf >> 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # [B, 64, 64, 64] --> [B,  128, 32, 32]
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf >> 4, ndf >> 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf >> 3),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # [B, 128, 32, 32] --> [B, 256, 16, 16]
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf >> 3, ndf >> 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf >> 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # [B, 256, 16, 16] --> [B, 512, 8, 8]
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf >> 2, ndf >> 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf >> 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # [B, 512, 8, 8] --> [B, 1024, 4, 4]
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf >> 1, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # [B, 1024, 4, 4] --> [B, 1, 1, 1]
        self.layer6 = nn.Sequential(
            nn.Conv2d(ndf, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):

        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.layer6(h)

        # [B, 1, 1, 1] --> [B, 1] --> [B]
        return h.view(-1, 1).squeeze(1)


if __name__ == "__main__":

    import torch

    netG = DCGANGenerator()
    print(netG)

    print(netG(torch.randn(10, 128)).shape)

    netD = DCGANDiscriminator()
    print(netD)

    print(netD(torch.randn(10, 3, 128, 128)).shape)

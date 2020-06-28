import math
import torch.nn as nn
import torch.nn.functional as F

from torch_mimicry.modules import SNConv2d
from layers import SBN, SCBN


class SGBlock(nn.Module):
    r"""
    Self modulated residual block for generator.

    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        num_classes (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
        nz (int): The dimension size of the input latent code, by default 128.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False,
                 num_classes=0,
                 spectral_norm=False,
                 nz=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample

        self.num_classes = num_classes
        self.spectral_norm = spectral_norm

        self.nz = nz

        # build the layers
        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels,
                               self.hidden_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
            self.c2 = SNConv2d(self.in_channels,
                               self.hidden_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        else:
            self.c1 = nn.Conv2d(self.in_channels,
                                self.hidden_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
            self.c2 = nn.Conv2d(self.hidden_channels,
                                self.out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        if self.num_classes == 0:
            self.b1 = SBN(self.in_channels, self.nz)
            self.b2 = SBN(self.hidden_channels, self.nz)
        else:
            self.b1 = SCBN(self.in_channels, self.num_classes, self.nz)
            self.b2 = SCBN(self.hidden_channels, self.num_classes, self.nz)

        self.activation = nn.ReLU(inplace=True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(self.in_channels,
                                     self.out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
            else:
                self.c_sc = nn.Conv2d(self.in_channels,
                                      self.out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _upsample(self, x, conv):
        r"""
        Helper function for performing convolution after upsampling.
        """

        return conv(
            F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )
        )

    def _residual(self, x, z):
        r"""
        Helper function for feedforwarding through main layers.
        """

        h = x
        h = self.activation(self.b1(h, z))
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.activation(self.b2(h, z))
        h = self.c2(h)

        return h

    def _residual_conditional(self, x, z, y):
        r"""
        Helper function for feedforwarding through main layers, including conditional BN.
        """

        h = x
        h = self.activation(self.b1(h, z, y))
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.activation(self.b2(h, z, y))
        h = self.c2(h)

        return h

    def _shortcut(self, x):
        r"""
        Helper function for feedforwarding through shortcut layers.
        """

        if self.learnable_sc:
            x = self._upsample_conv(
                x, self.c_sc
            ) if self.upsample else self.c_sc(x)

        return x

    def forward(self, x, z, y=None):
        r"""
        Residual block feedforward function.
        """

        if y is None:
            return self._residual(x, z) + self._shortcut(x)
        else:
            return self._residual_conditional(x, z, y) + self._shortcut(x)


if __name__ == "__main__":
    import torch

    B = 10
    nz = 128
    in_channels = 64
    out_channels = 128
    height = 32
    width = 32

    noise = torch.randn(B, nz)
    inputs = torch.randn(B, in_channels, height, width)

    # test Self modulated batch norm
    block = SGBlock(in_channels, out_channels, nz=nz)
    output = block(inputs, noise)
    print(output.shape)

    # test self modulated conditional batch norm
    num_classes = 10
    y = torch.randint(low=0, high=num_classes, size=(B,))

    block = SGBlock(in_channels, out_channels, num_classes=num_classes, nz=nz)
    output = block(inputs, noise, y)
    print(output.shape)

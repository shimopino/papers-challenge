"""
Script for building specific layers needed by GAN architecture.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_mimicry.modules.layers import SNConv2d


class SelfAttention(nn.Module):
    """
    Self-attention layer based on version used in BigGAN code:
    https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
    """
    def __init__(self, num_feat, spectral_norm=True):
        super().__init__()
        self.num_feat = num_feat

        if spectral_norm:
            self.f = SNConv2d(self.num_feat,
                              self.num_feat >> 3,
                              1,
                              1,
                              padding=0,
                              bias=False)
            self.g = SNConv2d(self.num_feat,
                              self.num_feat >> 3,
                              1,
                              1,
                              padding=0,
                              bias=False)
            self.h = SNConv2d(self.num_feat,
                              self.num_feat >> 1,
                              1,
                              1,
                              padding=0,
                              bias=False)
            self.o = SNConv2d(self.num_feat >> 1,
                              self.num_feat,
                              1,
                              1,
                              padding=0,
                              bias=False)

        else:
            self.f = nn.Conv2d(self.num_feat,
                               self.num_feat >> 3,
                               1,
                               1,
                               padding=0,
                               bias=False)
            self.g = nn.Conv2d(self.num_feat,
                               self.num_feat >> 3,
                               1,
                               1,
                               padding=0,
                               bias=False)
            self.h = nn.Conv2d(self.num_feat,
                               self.num_feat >> 1,
                               1,
                               1,
                               padding=0,
                               bias=False)
            self.o = nn.Conv2d(self.num_feat >> 1,
                               self.num_feat,
                               1,
                               1,
                               padding=0,
                               bias=False)

        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        """
        Feedforward function. Implementation differs from actual SAGAN paper,
        see note from BigGAN:
        https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py#L142
        """
        # 1x1 convs to project input feature map
        f = self.f(x)
        g = F.max_pool2d(self.g(x), [2, 2])
        h = F.max_pool2d(self.h(x), [2, 2])

        # Reshape layers
        f = f.view(-1, self.num_feat >> 3, x.shape[2] * x.shape[3])
        g = g.view(-1, self.num_feat >> 3, x.shape[2] * x.shape[3] >> 2)
        h = h.view(-1, self.num_feat >> 1, x.shape[2] * x.shape[3] >> 2)

        # Compute attention map probabiltiies
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), -1)

        # Weigh output features by attention map
        o = self.o(
            torch.bmm(h, beta.transpose(1, 2)).view(-1, self.num_feat >> 1,
                                                    x.shape[2], x.shape[3]))

        return self.gamma * o + x


class DBlockDecoder(nn.Module):
    r"""
    Residual block for Unet-based discriminator Decoder part.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False,
                 spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample
        self.spectral_norm = spectral_norm

        # Build the layers
        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, padding=1)
            self.c2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1, padding=1)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, padding=1)
            self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1, padding=1)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, padding=0)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _upsample_conv(self, x, conv):
        r"""
        Helper function for performing convolution after upsampling.
        """
        return conv(
            F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False))

    def _residual(self, x):
        r"""
        Helper function for feedforwarding through main layers.
        """
        h = self.activation(x)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.activation(h)
        h = self.c2(h)

        return h

    def _shortcut(self, x):
        r"""
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self._upsample_conv(
                x, self.c_sc) if self.upsample else self.c_sc(x)

        return x

    def forward(self, x):
        r"""
        Residual block feedforward function.

        Args:
            x (Tensor): the concatenated input feature map from both encode and decoder
        """

        return self._residual(x) + self._shortcut(x)


if __name__ == "__main__":
    inputs = torch.randn(10, 128, 32, 32)

    # DBlockDecoder when upsample=False
    dblock = DBlockDecoder(128, 128, upsample=False)
    output = dblock(inputs)
    print(output.shape)

    # backward
    loss = F.mse_loss(output, torch.ones(10, 128, 32, 32))
    loss.backward()

    # DBlockDecoder when upsample=True
    dblock = DBlockDecoder(128, 128, upsample=True)
    output = dblock(inputs)
    print(output.shape)

    # backward
    loss = F.mse_loss(output, torch.ones(10, 128, 64, 64))
    loss.backward()

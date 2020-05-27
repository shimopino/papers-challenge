import math
import torch.nn as nn
import torch.nn.functional as F
from .modules import SNConv2d, ConditionalBatchNorm2d


class GBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        upsample=False,
        num_classes=0,
        spectral_norm=False,
    ):
        """
        Resblock for Generator to deepen the model.
        This module uses bilinear interpolation to upsample inputs

        Args:
            in_channels (int): The channel size of input feature map
            out_channels (int): The channel size of output feature map
            hidden_channels (int, optional): The channel size of hidden feature map. If None, this is equals to out_channels.
            upsample (bool, optional): If True, upsamples the input feature map. Defaults to False.
            num_classes (int, optional): If more than 0, uses conditional batch norm instead. Defaults to 0.
            spectral_norm (bool, optional): If True, uses spectral norm for convolutional layers. Defaults to False.
        """
        super(GBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else out_channels
        )
        self.learnable_shortcut_conv = (in_channels != out_channels) or upsample
        self.upsample = upsample
        self.num_classes = num_classes
        self.spectral_norm = spectral_norm

        if self.spectral_norm:
            self.conv1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.conv2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1, 1)
        else:
            self.conv1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.conv2 = nn.Co初めてOSSにissueたてたnv2d(self.hidden_channels, self.out_channels, 3, 1, 1)

        if self.num_classes == 0:
            self.bn1 = nn.BatchNorm2d(self.in_channels)
            self.bn2 = nn.BatchNorm2d(self.hidden_channels)
        else:
            self.cbn1 = ConditionalBatchNorm2d(self.in_channels, self.num_classes)
            self.cbn2 = ConditionalBatchNorm2d(self.hidden_channels, self.num_classes)

        self.activation = nn.ReLU(inplace=True)

        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_shortcut_conv:
            if self.spectral_norm:
                self.shortcut_conv = SNConv2d(self.in_channels, self.out_channels, 1, 1, 0)
            else:
                self.shortcut_conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
            nn.init.xavier_uniform_(self.shortcut_conv.weight.data, 1.0)

    def _upsample_conv(self, x, conv):

        return conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))

    def _residual(self, x, y=None):
        """
        The input feature map has to come just after previous layer convolution.
        """

        h = x
        h = self.bn1(h) if y is None else self.cbn1(h, y)
        h = self.activation(h)
        h = self._upsample_conv(h, self.conv1) if self.upsample else self.conv1(h)
        h = self.bn2(h) if y is None else self.cbn1(h, y)
        h = self.activation(h)
        h = self.conv2(h)

        return h

    def _shortcut(self, x):

        if self.learnable_sc:
            if self.upsample:
                x = self._upsample_conv(x, self.shortcut_conv)
            else:
                x = self.c_sc(x)

        return x

    def forward(self, x, y=None):

        return self._residual(x, y) + self._shortcut(x)


class DBlockOptimized(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spectral_norm=True
    ):
        """
        Resblock for the First Layer of Discriminator to definitely downsample inputs.
        Follow the Official SNGAN inplementation by chainer.

        Args:
            in_channels (int): The channel size of input feature map
            out_channels (int): The channel size of output feature map
            spectral_norm (bool, optional): If True, uses spectral norm for convolutional layers. Defaults to True.
        """
        super(DBlockOptimized, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm

        # Build Layers
        if self.spectral_norm:
            self.conv1 = SNConv2d(self.in_channels, self.out_channels, 3, 1, 1)
            self.conv2 = SNConv2d(self.out_channels, self.out_channels, 3, 1, 1)
            self.shortcut_conv = SNConv2d(self.in_channels, self.out_channels, 1, 1, 0)
        else:
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
            self.shortcut_conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.shortcut_conv.weight.data, math.sqrt(2.0))

    def _residual(self, x):

        h = x
        h = self.conc1(h)
        h = self.activation(h)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):

        return self.shortcut_conv(F.avg_pool2d(x, 2))

    def forward(self, x):

        return self._residual(x) + self._shortcut(x)


class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        downsample=False,
        spectral_norm=True,
    ):
        """
        Resblock for Discriminator to deepen the layers.

        Args:
            in_channels (int): The channel size of input feature map
            out_channels (int): The channel size of input feature map
            hidden_channels (int, optional): The channel size of input feature map. If None, this is equal to out_channels Defaults to None.
            downsample (bool, optional): If True, downsample the input feature map. Defaults to False.
            spectral_norm (bool, optional): If True, uses spectral norm for convolutional layers. Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else in_channels
        )
        self.downsample = downsample
        self.learnable_shortcut_conv = (in_channels != out_channels) or downsample
        self.spectral_norm = spectral_norm

        if self.spectral_norm:
            self.conv1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.conv2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1, 1)
        else:
            self.conv1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1, 1)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_shortcut_conv:
            if self.spectral_norm:
                self.shortcut_conv = SNConv2d(self.in_channels, self.out_channels, 1, 1, 0)
            else:
                self.shortcut_conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
            nn.init.xavier_uniform_(self.shortcut_conv.weight.data, 1.0)

    def _residual(self, x):

        h = x
        h = self.activation(h)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):

        if self.learnable_shortcut_conv:
            x = self.shortcut_conv(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)

        return x

    def forward(self, x):

        return self._residual(x) + self._shortcut(x)


if __name__ == "__main__":
    import torch

    gblock = GBlock(32, 64, 64, False, 0, True)
    output = gblock(torch.randn(10, 32, 64, 64))
    print(output.shape)

    doptblock = DBlockOptimized(3, 32, True)
    output = doptblock(torch.randn(10, 3, 64, 64))
    print(output.shape)

    dblock = DBlock(32, 64, 64, False, True)
    output = dblock(torch.randn(10, 32, 64, 64))
    print(output.shape)

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
        use_sn=False,
    ):
        super(GBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else out_channels
        )
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample
        self.num_classes = num_classes
        self.use_sn = use_sn

        self.c1 = SNConv2d(use_sn, self.in_channels, self.hidden_channels, 3, 1, 1)
        self.c2 = SNConv2d(use_sn, self.hidden_channels, self.out_channels, 3, 1, 1)

        if self.num_classes == 0:
            self.b1 = nn.BatchNorm2d(self.in_channels)
            self.b2 = nn.BatchNorm2d(self.hidden_channels)
        else:
            self.b1 = ConditionalBatchNorm2d(self.in_channels, self.num_classes)
            self.b2 = ConditionalBatchNorm2d(self.hidden_channels, self.num_classes)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            self.c_sc = SNConv2d(use_sn, self.in_channels, self.out_channels, 1, 1, 0)
            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _upsample_conv(self, x, conv):

        return conv(
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        )

    def _residual(self, x, y=None):

        h = x
        h = self.b1(h) if y is None else self.b1(h, y)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h) if y is None else self.b1(h, y)
        h = self.activation(h)
        h = self.c2(h)

        return h

    def _shortcut(self, x):

        if self.learnable_sc:
            x = self._upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)

        return x

    def forward(self, x, y=None):

        return self._residual(x, y) + self._shortcut(x)


class DBlockOptimized(nn.Module):
    def __init__(self, in_channels, out_channels, use_sn=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_sn = use_sn

        # Build Layers
        self.c1 = SNConv2d(use_sn, self.in_channels, self.out_channels, 3, 1, 1)
        self.c2 = SNConv2d(use_sn, self.out_channels, self.out_channels, 3, 1, 1)
        self.c_sc = SNConv2d(use_sn, self.in_channels, self.out_channels, 1, 1, 0)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):

        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):

        return self.c_sc(F.avg_pool2d(x, 2))

    def forward(self, x):

        return self._residual(x) + self._shortcut(x)


class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        downsample=False,
        use_sn=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else in_channels
        )
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.use_sn = use_sn

        self.c1 = SNConv2d(use_sn, self.in_channels, self.hidden_channels, 3, 1, 1)
        self.c2 = SNConv2d(use_sn, self.hidden_channels, self.out_channels, 3, 1, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            self.c_sc = SNConv2d(use_sn, self.in_channels, self.out_channels, 1, 1, 0)
            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):

        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):

        if self.learnable_sc:
            x = self.c_sc(x)
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

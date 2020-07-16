import torch.nn as nn
from torch_mimicry.modules import SNConv2d
from .modules import ConditionalBatchNorm2d_with_skip_and_shared


class DBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 downsample=False,
                 spectral_norm=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels \
                               if hidden_channels is not None else in_channels
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.spectral_norm = spectral_norm

        if self.spectral_norm:
           self.c1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
           self.c2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1, 1)
        else:
           self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
           self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1, 1)

        self.average_pooling = nn.AvgPool2d(2)
        self.activation = nn.ReLU(inplace=True)

        # Shortcut layer
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            
    def forward(self, x):
        x0 = x

        # shortcut branch
        if self.learnable_sc:
            x0 = self.c_sc(x0)
            if self.downsample:
                x0 = self.average_pooling(x0, 2)

        # normal branch
        x = self.activation(x)
        x = self.c1(x)
        x = self.activation(x)
        x = self.c2(x)
        if self.downsample:
            x = self.average_pooling(x, 2)

        # residual connection
        out = x + x0
        return out


class GBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False,
                 concat_vector_dim=None,
                 spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample

        self.concat_vector_dim= concat_vector_dim
        self.spectral_norm = spectral_norm

        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.c2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1, 1)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1, 1)

        if self.num_classes == 0:
            self.b1 = nn.BatchNorm2d(self.in_channels)
            self.b2 = nn.BatchNorm2d(self.hidden_channels)
        else:
            self.b1 = ConditionalBatchNorm2d_with_skip_and_shared(
                         self.in_channels,
                         self.concat_vector_dim
                      )
            self.b2 = ConditionalBatchNorm2d_with_skip_and_shared(
                         self.hidden_channels,
                         self.concat_vector_dim
                      )

        self.activation = nn.ReLU(True)

        # Shortcut layer
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, concat_vector):
        
        x0 = x

        # shortcut branch
        if self.upsample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        if self.learnable_sc:
            x0 = self.c_sc(x0)

        # normal branch
        ## first block
        if self.concat_vector_dim is not None:
            x = self.b1(x, concat_vector)
        else:
            x = self.b1(x)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        ## second block
        if self.concat_vector_dim is not None:
            x = self.b2(x, concat_vector)
        else:
            x = self.b2(x)
        x = self.activation(x)
        x = self.c2(x)

        # residual connection
        out = x + x0
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mimicry.modules import SpectralNorm


class SNConvTranspose2dBase(nn.ConvTranspose2d, SpectralNorm):
    r"""
    Spectrally normalized layer for ConvTranspose2d.

    Attributes:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimensions.
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, *args, **kwargs)

        SpectralNorm.__init__(self,
                              n_dim=out_channels,
                              num_iters=kwargs.get('num_iters', 1))

    def forward(self, x):
        return F.conv_transpose2d(input=x,
                                  weight=self.sn_weights(),
                                  bias=self.bias,
                                  stride=self.stride,
                                  padding=self.padding,
                                  output_padding=self.output_padding,
                                  dilation=self.dilation,
                                  groups=self.groups)


def SNConvTranspose2d(*args, **kwargs):
    r"""
    Wrapper for applying spectral norm on linear layer.
    """
    if kwargs.get('default', True):
        return nn.utils.spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

    else:
        return SNConvTranspose2dBase(*args, **kwargs)


def ortho_reg(model, strength=1e-4, blacklist=None):
    """
    Apply Orthogonal Regularization after calculating gradient using loss backward().

    Args:
        model (nn.Module): nn.Module Model after loss backward
        strength (float, optional): parameter for strengthing the effect of this regularization. Defaults to 1e-4.
        blacklist (list, optional): set to avoid to regulate shared Generator layers. Defaults to None.
    """

    # to avoid iterable error because Python’s default arguments are evaluated once
    # when the function is defined, not each time the function is called.
    if blacklist is None:
        blacklist = []

    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes, and not in the blacklist
            if (len(param.shape) < 2) or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = 2 * torch.mm(
                torch.mm(w, w.t()) * (1.0 - torch.eye(w.shape[0], device=w.device)), w
            )
            param.grad.data += strength * grad.view(param.shape)

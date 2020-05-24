import torch
import torch.nn as nn


def SNConv2d(use_sn, *args, **kwargs):
    if use_sn:
        return torch.nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
    else:
        return nn.Conv2d(*args, **kwargs)


def SNLinear(use_sn, *args, **kwargs):
    if use_sn:
        return torch.nn.utils.spectral_norm(nn.Linear(*args, **kwargs))
    else:
        return nn.Linear(*args, **kwargs)


def SNEmbedding(use_sn, *args, **kwargs):
    if use_sn:
        return torch.nn.utils.spectral_norm(nn.Embedding(*args, **kwargs))
    else:
        return nn.Embedding(*args, **kwargs)


def SNConvTranspose2d(use_sn, *args, **kwargs):
    if use_sn:
        return torch.nn.utils.spectral_norm(nn.ConvTranspose2d(*args, **kwargs))
    else:
        return nn.ConvTranspose2d(*args, **kwargs)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight_data[:, :num_features].normal_(1, 0.02)
        self.embed.weight_data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1
        )

        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

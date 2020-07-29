import torch.nn as nn
from torch_mimicry.modules import SNLinear


class ConditionalBatchNorm2d_with_skip_and_shared(nn.Module):
    def __init__(self, num_features, concat_vector_dim, spectral_norm=False):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)

        if spectral_norm:
            self.gain = SNLinear(concat_vector_dim, num_features, bias=False)
            self.bias = SNLinear(concat_vector_dim, num_features, bias=False)
        else:
            self.gain = nn.Linear(concat_vector_dim, num_features, bias=False)
            self.bias = nn.Linear(concat_vector_dim, num_features, bias=False)

    def forward(self, x, concat_vector):
        r"""feed-forward the input feature map and concatenated embedding

        Args:
            x (Tensor): the input feature map of shape [B, C, H, W]
            concat_vector (Tensor): concatenated vector of latent code and class embedding
        """
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1) # one-centered gain
        bias = self.bias(y).view(y.size(0), -1, 1, 1) # zero-centered bias
        out = self.bn(x)
        return out * gain + bias

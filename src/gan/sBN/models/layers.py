import torch.nn as nn


class SBN(nn.Module):
    r"""
    Self Modulated Generative Adversarial Networks
    https://arxiv.org/abs/1810.01365

    Attributes:
        num_features (int): The channel size of the input feature map.
        nz (int): The dimension size of the latent code.
        hidden_features (int): The dimension size of the hidden latent code for gamma & beta, by default 32.
    """
    def __init__(self,
                 num_features,
                 nz,
                 hidden_features=32):
        super().__init__()
        self.num_features = num_features

        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.z_to_h = nn.Linear(nz, hidden_features)
        self.h_to_gamma = nn.Linear(hidden_features, num_features)
        self.h_to_beta = nn.Linear(hidden_features, num_features)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, z):
        r"""
        Feedforwards for self modulated batch norm

        Args:
            x (torch.Tensor): Input Feature Map
            z (torch.Tensor): Input Latent Code

        Returns:
            Tensor: Output feature map
        """
        x_norm = self.bn(x)
        h = self.activation(self.z_to_h(z))
        gamma = self.h_to_gamma(h).view(-1, self.num_features, 1, 1)
        beta = self.h_to_beta(h).view(-1, self.num_features, 1, 1)
        output = gamma + x_norm + beta
        return output


class SCBN(nn.Module):
    r"""
    Self Modulated Generative Adversarial Networks
    https://arxiv.org/abs/1810.01365

    Attributes:
        num_features (int): The channel size of the input feature map.
        num_classes (int): Determines size of embedding layer to condition BN.
        nz (int): The dimension size of the latent code.
        hidden_features (int): The dimension size of the hidden latent code for gamma & beta, by default 32.
    """
    def __init__(self,
                 num_features,
                 num_classes,
                 nz,
                 hidden_features=32):
        super().__init__()
        self.num_features = num_features

        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, nz * 2)
        self.z_to_h = nn.Linear(nz, hidden_features)
        self.h_to_gamma = nn.Linear(hidden_features, num_features)
        self.h_to_beta = nn.Linear(hidden_features, num_features)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, z, y):
        x_norm = self.bn(x)
        embed_y, embed_y_prime = self.embed(y).chunk(2, 1)
        z_prime = z + embed_y + z * embed_y_prime
        h = self.activation(self.z_to_h(z_prime))
        gamma = self.h_to_gamma(h).view(-1, self.num_features, 1, 1)
        beta = self.h_to_beta(h).view(-1, self.num_features, 1, 1)
        output = gamma + x_norm + beta
        return output


if __name__ == "__main__":
    import torch

    B = 10
    nz = 128
    num_features = 64
    height = 32
    width = 32

    noise = torch.randn(B, nz)
    inputs = torch.randn(B, num_features, height, width)

    # self modulated batch norm
    sBN = SBN(num_features, nz, hidden_features=32)
    output = sBN(inputs, noise)
    print(output.shape)

    # self modulated conditional batch norm
    num_classes = 10
    y = torch.randint(low=0, high=num_classes, size=(B,))

    cSBN = SCBN(num_features, num_classes, nz, hidden_features=32)
    output = cSBN(inputs, noise, y)
    print(output.shape)

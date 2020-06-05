import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureQuantizer(nn.Module):
    """
    Feature Quantization Modules toward the Output Feature Maps of Discriminator.
    This Modules follow the Equation (7) in the original paper.

    Args:
        emb_dim (int): the number of dimensions of Embedding Vector
        num_emb (int): the power of the dictionary size
        commitment (float, optional): the strength of commitment loss. Defaults to 0.25.
    """
    def __init__(self, emb_dim, num_emb, commitment=0.25):
        super(FeatureQuantizer, self).__init__()

        self.emb_dim = emb_dim
        self.num_emb = num_emb
        self.commitment = commitment
        self.embed = nn.Parameter(torch.randn(emb_dim, num_emb))

    def forward(self, inputs):
        # [B, C=D, H, W] --> [B, H, W, C=D]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs_shape = inputs.size()

        # flatten: [B, H, W, C=D] --> [BxHxW=N, D]
        flatten = inputs.view(-1, self.emb_dim)

        # distance: d(W[N, D], E[D, K]) <-- [N, K]
        distance = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        # minimum index: [N, K] --> [N, ]
        embed_idx = torch.argmin(distance, dim=1)

        # set OneHot label: [N, ] --> [N, K]
        embed_onehot = F.one_hot(embed_idx, num_classes=self.num_emb).type(
            flatten.dtype
        )

        # quantize: [N, ] --> [B, H, W, ]
        embed_idx = embed_idx.view(*inputs_shape[:-1])
        # quantize: [B, H, W, ] embed [K, D] --> [B, H, W, D]
        quantize = F.embedding(embed_idx, self.embed.transpose(0, 1))

        # loss
        e_latent_loss = F.mse_loss(quantize.detach(), inputs)
        # e_latent_loss = (quantize.detach() - inputs).pow(2).mean()
        q_latent_loss = F.mse_loss(quantize, inputs.detach())
        loss = q_latent_loss + self.commitment * e_latent_loss

        # Straight Through Estimator
        quantize = inputs + (quantize - inputs).detach()
        # [B, H, W, D] --> [B, D, H, W]
        # quantize = quantize.permute(0, 3, 1, 2).contiguous()

        # average probability: [N, K] --> [N, ]
        # avg_probs = torch.mean(embed_onehot, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantize.permute(0, 3, 1, 2).contiguous(), loss, embed_onehot

    def extra_repr(self):
        return "emb_dim={}, num_emb={}, commitment={}".format(
            self.emb_dim, self.num_emb, self.commitment
        )


class FeatureQuantizerEMA(nn.Module):
    """
    Feature Quantization Modules using Exponential Moving Average toward the Output Feature Maps of Discriminator.
    This Modules follow the Equation (8) in the original paper.

    Args:
        emb_dim (int): the number of dimensions of Embedding Vector
        num_emb (int): the power of the dictionary size
        commitment (float, optional): the strength of commitment loss. Defaults to 0.25.
        decay (float, optional): the moment coefficient. Defaults to 0.9.
        eps (float, optional): the sufficient small value to avoid dividing by zero. Defaults to 1e-5.
    """
    def __init__(self, emb_dim, num_emb, commitment=0.25, decay=0.9, eps=1e-5):
        super(FeatureQuantizerEMA, self).__init__()

        self.emb_dim = emb_dim
        self.num_emb = num_emb
        self.commitment = commitment
        self.decay = decay
        self.eps = eps

        embed = torch.randn(emb_dim, num_emb)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_emb))
        self.register_buffer("ema_embed", embed.clone())

    def forward(self, inputs):
        # [B, C=D, H, W] --> [B, H, W, C=D]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs_shape = inputs.size()

        # flatten: [B, H, W, C=D] --> [BxHxW=N, D]
        flatten = inputs.view(-1, self.emb_dim)

        # distance: d(W[N, D], E[D, K]) <-- [N, K]
        distance = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        # minimum index: [N, K] --> [N, ]
        embed_idx = torch.argmin(distance, dim=1)

        # set OneHot label: [N, ] --> [N, K]
        embed_onehot = F.one_hot(embed_idx, num_classes=self.num_emb).type(
            flatten.dtype
        )

        # quantize: [N, ] --> [B, H, W, ]
        embed_idx = embed_idx.view(*inputs_shape[:-1])
        # quantize: [B, H, W, ] embed [K, D] --> [B, H, W, D]
        quantize = F.embedding(embed_idx, self.embed.transpose(0, 1))

        # train embedding vector only when model.train() not model.eval()
        if self.training:
            # ref_counts: [N, K] --> [K, ]
            ref_counts = torch.sum(embed_onehot, dim=0)

            # ema for reference counts: [K, ]
            self.cluster_size.data.mul_(self.decay).add_(
                ref_counts, alpha=1 - self.decay
            )

            # total reference count
            n = self.cluster_size.sum()

            # laplace smoothing
            smoothed_cluster_size = n * (
                (self.cluster_size + self.eps) / (n + self.cluster_size * self.eps)
            )

            # dw: [D, N] @ [N, K]
            dw = flatten.transpose(0, 1) @ embed_onehot

            # ema for embeddings: [D, K]
            self.ema_embed.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # normalize by reference counts: [D, K] / [1, K] <-- [K, ]
            embed_normalized = self.ema_embed / smoothed_cluster_size.unsqueeze(0)
            # self.embed = self.ema_embed / self.cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # loss
        # This is equivalent to TensorFlow's stop gradient
        # tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        e_latent_loss = F.mse_loss(quantize.detach(), inputs)
        loss = self.commitment * e_latent_loss

        # Straight Through Estimator
        quantize = inputs + (quantize - inputs).detach()
        # [B, H, W, D] --> [B, D, H, W]
        # quantize = quantize.permute(0, 3, 1, 2).contiguous()

        # average probability: [N, K] --> [N, ]
        # avg_probs = torch.mean(embed_onehot, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantize.permute(0, 3, 1, 2).contiguous(), loss, embed_idx

    def extra_repr(self):
        return "emb_dim={}, num_emb={}, commitment={}, decay={}, eps={}".format(
            self.emb_dim, self.num_emb, self.commitment, self.decay, self.eps
        )


if __name__ == "__main__":

    emb_dim = 256
    num_emb = 2 ** 10

    # check forward propagation of VectorQuantizer
    vq = FeatureQuantizer(emb_dim, num_emb)
    print(vq)
    inputs = torch.randn(100, emb_dim, 64, 64)

    vq.train()
    h, loss, embed_idx = vq(inputs)
    assert h.shape == inputs.shape
    print(h.shape)

    # check forward propagation of VectorQuantizerEMA
    vqema = FeatureQuantizerEMA(emb_dim, num_emb)
    print(vqema)
    inputs = torch.randn(100, emb_dim, 64, 64)

    vqema.train()
    h, loss, embed_idx = vqema(inputs)
    assert h.shape == inputs.shape
    print(h.shape)

    # check inference forward propagation of VectorQuantizerEMA
    vqema = FeatureQuantizerEMA(emb_dim, num_emb)
    inputs = torch.randn(100, emb_dim, 64, 64)

    vqema.eval()
    h, loss, embed_idx = vqema(inputs)
    assert h.shape == inputs.shape
    print(h.shape)

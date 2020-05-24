import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, emb_dim, num_emb, commitment=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self.emb_dim = emb_dim
        self.num_emb = num_emb
        self.commitment = commitment
        self.decay = decay
        self.epsilon = epsilon

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
        quantize = quantize.view(*inputs_shape)

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
                (self.cluster_size + self.epsilon)
                / (n + self.cluster_size * self.epsilon)
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
        e_latent_loss = F.mse_loss(quantize.detach(), inputs)
        # e_latent_loss = (quantize.detach() - inputs).pow(2).mean()
        loss = self.commitment * e_latent_loss

        # Straight Through Estimator
        quantize = inputs + (quantize - inputs).detach()
        # average probability: [N, K] --> [N, ]
        avg_probs = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantize.permute(0, 3, 1, 2).contiguous(), loss, perplexity


if __name__ == "__main__":
    vq = VectorQuantizerEMA(32, 2 ** 5)
    inputs = torch.randn(100, 32, 64, 64)

    vq.train()
    h, loss, ppl = vq(inputs)
    print(h.shape)

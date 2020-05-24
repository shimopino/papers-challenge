import torch
import torch.nn as nn


class ProbLoss(nn.Module):
    def __init__(self, loss_type, batch_size, device):
        assert loss_type in ["bce", "hinge"]
        super().__init__()
        self.loss_type = loss_type
        self.device = device
        self.ones = torch.ones(batch_size).to(device)
        self.zeros = torch.zeros(batch_size).to(device)
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, logits, condition):
        assert condition in ["gen", "dis_real", "dis_fake"]
        batch_len = len(logits)

        if self.loss_type == "bce":
            if condition in ["gen", "dis_real"]:
                return self.bce(logits, self.ones[:batch_len])
            else:
                return self.bce(logits, self.zeros[:batch_len])

        elif self.loss_type == "hinge":
            if condition == "gen":
                return -torch.mean(logits)
            elif condition == "dis_real":
                minval = torch.min(logits - 1, self.zeros[:batch_len])
                return -torch.mean(minval)
            else:
                minval = torch.min(-logits - 1, self.zeros[:batch_len])
                return -torch.mean(minval)


def ortho_reg(model, factor=1e-4, device=None):
    with torch.enable_grad():
        ortho_loss = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if "bias" not in name:
                w = param.view(param.size(0), -1)
                sym = (w @ w.t()) * (1.0 - torch.eye(w.size(0), device=device))
                ortho_loss += factor * sym.abs().sum()

    return ortho_loss

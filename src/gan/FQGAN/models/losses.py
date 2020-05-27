import torch
import torch.nn as nn


class ProbLoss(nn.Module):
    def __init__(self, loss_type, batch_size, device):
        assert loss_type in ["bce", "hinge"]
        super().__init__()
        self.loss_type = loss_type
        self.device = device
        self.ones = torch.ones(batch_size, dtype=torch.float, device=device)
        self.zeros = torch.zeros(batch_size, dtype=torch.float, device=device)
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
                minval = torch.min(logits - 1.0, self.zeros[:batch_len])
                return -torch.mean(minval)
            else:
                minval = torch.min(-logits - 1.0, self.zeros[:batch_len])
                return -torch.mean(minval)


def ortho_reg(model, strength=1e-4, blacklist=[]):
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

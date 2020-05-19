import torch


def orthogonal_regularization(model, factor=1e-4):
    with torch.no_grad():
        ortho_loss = torch.tensor(0.)
        for name, param in model.named_parameters():
            if "bias" not in name:
                w = param.view(param.size(0), -1)
                sym = ((w @ w.t()) * (1. - torch.eye(w.size(0), device=device))).pow(2)
                ortho_loss += factor * sym.sum()

    return ortho_loss
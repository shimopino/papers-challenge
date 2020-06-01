import torch


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

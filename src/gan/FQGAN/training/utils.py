import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def count_parameters_float32(model):
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_mb = param / 1024 / 1024 / 8 * 32
    return param_mb

import torch
import torch.nn as nn


class ProbLoss(nn.Module):
    def __init__(self, cfg):
        assert cfg.loss_type in ["bce", "hinge"]
        super().__init__()
        self.loss_type = cfg.loss_type
        self.device = cfg.device
        self.ones = torch.ones(cfg.batch_size).to(cfg.device)
        self.zeros = torch.zeros(cfg.batch_size).to(cfg.device)
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

"""
Dice loss 3D
"""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        num_vec = 2*torch.einsum('bcijk, bcijk ->bc',
                [preds, targets])
        denom = torch.einsum('bcijk, bcijk -> bc',
                [preds, preds]) +\
                torch.einsum('bcijk, bcijk -> bc',
                        [targets, targets])
        proportions = torch.div(num_vec, denom)

        avg_dice = torch.einsum('bc->', proportions) / (targets.shape[0]*targets.shape[1])
        return 1 - avg_dice

class Recon(nn.Module):
    def __init__(self):
        super(Recon, self).__init__()

    def forward(self, pred, target):
        pass

class DiceRecon(nn.Module):
    pass

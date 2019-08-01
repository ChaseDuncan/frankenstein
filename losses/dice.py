"""
Dice loss 3D
"""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()


    def forward(self, preds, targets):
        num_vec = 2*torch.einsum('cijk, cijk ->c', 
                [preds.squeeze(0), targets.squeeze(0)])
        denom = torch.einsum('cijk, cijk -> c', 
                [preds.squeeze(0), preds.squeeze(0)]) +\
                torch.einsum('cijk, cijk -> c', 
                        [targets.squeeze(0), targets.squeeze(0)])
        avg_dice = torch.sum(num_vec / denom) / targets.squeeze(0).shape[0]
        return 1 - avg_dice

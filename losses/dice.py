"""
Dice loss 3D
"""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()


    def forward(self, preds, targets):
        ''' Expect preds and targets to each be 3xHxWxD.''' 
        num_vec = 2*torch.einsum('cijk, cijk ->c', 
                [preds.squeeze(), targets.squeeze()])
        denom = torch.einsum('cijk, cijk -> c', 
                [preds.squeeze(), preds.squeeze()]) +\
                torch.einsum('cijk, cijk -> c', 
                        [targets.squeeze(), targets.squeeze()])
        avg_dice = torch.sum(num_vec / denom) / 3.0
        if 1 - avg_dice < 0:
            import pdb; pdb.set_trace()
        return 1 - avg_dice

"""
Dice loss 3D
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class AvgDiceLoss(nn.Module):
  def __init__(self):
    super(AvgDiceLoss, self).__init__()

  def forward(self, preds, targets):
    num_vec = 2*torch.einsum('bcijk, bcijk ->bc', [preds, targets])
    denom = torch.einsum('bcijk, bcijk -> bc', [preds, preds]) +\
        torch.einsum('bcijk, bcijk -> bc', [targets, targets])
    proportions = torch.div(num_vec, denom)

    avg_dice = torch.einsum('bc->', proportions) / (targets.shape[0]*targets.shape[1])
    return 1 - avg_dice

# TODO: Should batch be in here?
class DiceLoss(nn.Module):
  def __init__(self):
    super(DiceLoss, self).__init__()

  def forward(self, preds, targets):
    num_vec = 2*torch.einsum('bcijk, bcijk ->bc', [preds, targets])
    denom = torch.einsum('bcijk, bcijk -> bc', [preds, preds]) +\
            torch.einsum('bcijk, bcijk -> bc', [targets, targets])
    proportions = torch.div(num_vec, denom)

    return torch.einsum('bc->', proportions)

class Recon(nn.Module):
  def __init__(self):
    super(Recon, self).__init__()

  def forward(self, pred, target):
    return F.mse_loss(pred, target)



class DiceRecon(nn.Module):
  def __init__(self):
    super(DiceRecon, self).__init__()
    self.dice = AvgDiceLoss()
    self.recon = Recon()

  def forward(self, preds, targets):
    dice_loss = self.dice(preds[:, :-1, :, :, :], targets[:, :-1, :, :, :])
    recon_loss = self.recon(preds[:, -1, :, :, :], targets[:, -1, :, :, :])
    return dice_loss + 0.1*recon_loss

"""
Cross Entropy 3D
"""

class CrossEntropyLoss(nn.Module):
  def __init__(self):
    super(CrossEntropyLoss, self).__init__()
    self.loss = nn.CrossEntropyLoss(ignore_index=0, size_average=True)

  def forward(self, inputs, targets):
    return self.loss(inputs, targets)

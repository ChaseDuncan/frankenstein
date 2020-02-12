"""
Dice loss 3D
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

def dice_score(preds, targets):
  # TODO: the way this function is implemented and used assumes
  # a batch size of 1. This could cause bugs.
  if isinstance(preds, tuple):
    preds = preds[0]
  num = 2*torch.einsum('bcijk, bcijk ->bc', [preds, targets])
  denom = torch.einsum('bcijk, bcijk -> bc', [preds, preds]) +\
      torch.einsum('bcijk, bcijk -> bc', [targets, targets]) + 1e-9
  proportions = torch.div(num, denom) 
  return torch.einsum('bc->c', proportions)

def agg_dice_score(preds, targets):
  ''' Gives Dice score for sub-regions which are evaluated in the
  competition.
  '''
  if isinstance(preds, tuple):
    preds = preds[0]

  channel_shape = preds[:, 0, :, :, :].size()

  agg = torch.zeros(preds.size())
  et = torch.zeros(channel_shape)
  et[torch.where(preds[:, 2, :, :, :] > 0.5)] = 1
  et = et.unsqueeze(1)
  
  wt = torch.zeros(channel_shape)
  wt[torch.where((preds[:, 0, :, :, :] > 0.5) | (preds[:, 1, :, :, :] > 0.5) | (preds[:, 2, :, :, :] > 0.5) )] = 1
  wt = wt.unsqueeze(1)

  tc = torch.zeros(channel_shape)
  tc[torch.where((preds[:, 0, :, :, :] > 0.5) | (preds[:, 2, :, :, :] > 0.5) )] = 1
  tc = tc.unsqueeze(1)
  
  agg_preds = torch.cat((et, wt, tc), 1)

  et_target = torch.zeros(channel_shape)
  et_target[torch.where(targets[:, 2, :, :, :] > 0.5)] = 1
  et_target = et_target.unsqueeze(1) 

  wt_target = torch.zeros(channel_shape)
  wt_target[torch.where((targets[:, 0, :, :, :] > 0.5) | (targets[:, 1, :, :, :] > 0.5) | (targets[:, 2, :, :, :] > 0.5) )] = 1
  wt_target = wt_target.unsqueeze(1)
  
  tc_target = torch.zeros(channel_shape)
  tc_target[torch.where((targets[:, 0, :, :, :] > 0.5) | (targets[:, 2, :, :, :] > 0.5) )] = 1
  tc_target = tc_target.unsqueeze(1)

  agg_targets = torch.cat((et_target, wt_target, tc_target), 1)

  return dice_score(agg_preds, agg_targets)

class KLLoss(nn.Module):
  def __init__(self):
    super(KLLoss, self).__init__()

  def forward(self, mu, logvar, N):
    sum_square_mean = torch.einsum('i,i->', mu, mu)
    sum_log_var = torch.einsum('i->', logvar)
    sum_var = torch.einsum('i->', torch.exp(logvar))
    
    return float(1/N)*(sum_square_mean+sum_var-sum_log_var-N)


class VAEDiceLoss(nn.Module):
  def __init__(self, label_recon = False):
    super(VAEDiceLoss, self).__init__()
    self.dice = AvgDiceLoss()
    self.kl = KLLoss()
    self.label_recon = label_recon

  def forward(self, outputs_targets):
    output, targets, src = outputs_targets
    preds, recon, mu, logvar = output
    if self.label_recon:
       self.dice(preds, targets, src) + 0.1*F.mse_loss(recon, targets) + 0.1*self.kl(mu, logvar, 256)

    return self.dice(preds, targets, src) + 0.1*F.mse_loss(recon, src) + 0.1*self.kl(mu, logvar, 256)


class AvgDiceLoss(nn.Module):
  def __init__(self):
    super(AvgDiceLoss, self).__init__()
  # Need a loss builder so we don't have to have superfluous arguments

  def forward(self, preds, targets, src):
    proportions = dice_score(preds, targets)
    avg_dice = torch.einsum('c->', proportions) / (targets.shape[0]*targets.shape[1])
    return 1 - avg_dice


class DiceLoss(nn.Module):
  def __init__(self):
    super(DiceLoss, self).__init__()

  def forward(self, preds_and_targets):
    preds, targets, _ = preds_and_targets
    num_channels = targets.size()[1]
    return  num_channels - torch.einsum('c->', dice_score(preds, targets))

class ReconRegLoss(nn.Module):
  def __init__(self):
    super(ReconRegLoss, self).__init__()
    self.dice = AvgDiceLoss()

  def forward(self, output_targets):
    output, target, src = output_targets
    preds, recon = output
    dice_loss = self.dice(preds, target, src)
    return dice_loss + 0.1*F.mse_loss(recon, src)

def build(config):
  if config.loss == 'dice':
    return DiceLoss()
  if config.loss == 'recon':
    return ReconRegLoss()
  if config.loss == 'avgdice':
    return AvgDiceLoss()
  if config.loss == 'vae':
    if config.label_recon:
     return VAEDiceLoss(label_recon=config.label_recon) 
    return VAEDiceLoss()

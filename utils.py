import torch

def dice_score(preds, targets):
    num_vec = 2*torch.einsum('cijk, cijk ->c', \
            [preds.squeeze(), targets.squeeze()])
    denom = torch.einsum('cijk, cijk -> c', \
            [preds.squeeze(), preds.squeeze()]) +\
                torch.einsum('cijk, cijk -> c', \
                [targets.squeeze(), targets.squeeze()])
    avg_dice = num_vec / denom
    return avg_dice



"""
Cross Entropy 3D
"""

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=0, size_average=True)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)

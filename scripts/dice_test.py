from losses.dice import dice_score
import numpy as np
import torch

src = np.random.randint(2, size=(1, 1, 3, 3, 3))
tgt = np.random.randint(2, size=(1, 1, 3, 3, 3))
print(src)
print(tgt)

src = torch.from_numpy(src)
tgt = torch.from_numpy(tgt)
print(dice_score(src, tgt))


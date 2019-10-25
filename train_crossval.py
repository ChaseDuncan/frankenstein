import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler
import numpy as np
import pickle
import argparse

from utils import (
        dice_score, 
        MRISegConfigParser,
        save_model,
        cross_validation,
        train
        )
from torch.utils.data import DataLoader
from losses.dice import DiceLoss
from model.btseg import BraTSSegmentation
from datasets.data_loader import BraTSDataset

random_seed = 0

parser = argparse.ArgumentParser(description='Train MRI segmentation model.')
parser.add_argument('--config')
args = parser.parse_args()

config = MRISegConfigParser(args.config)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

brats_data = BraTSDataset(config.data_dir, config.labels, modes=config.modes)

# TODO: Make checkpoints dir if doesn't exist

cv_trainloaders, cv_testloaders = \
        cross_validation(brats_data, batch_size=1, 
                deterministic_train=config.deterministic_train, random_seed=random_seed)

# Fix stochasticity in model params, etc.
for i, (trainloader, testloader) in enumerate(zip(cv_trainloaders, cv_testloaders)):
    if config.deterministic_train:
        torch.manual_seed(random_seed)

    input_channels = len(config.modes)
    output_channels = len(config.labels)

    model = BraTSSegmentation(input_channels, output_channels) 
    model = model.to(device)
    import pdb; pdb.set_trace()
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1)
    #optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
    model_name = "{}_fold_{}".format(config.model_name, i)
    optimizer = \
            optim.Adam(model.parameters(), lr=1e-4, weight_decay=config.weight_decay)
    train(model, DiceLoss(), optimizer, trainloader, testloader, config.max_epochs, device,
        name=model_name, best_eval=0.0, epoch=0)
    # TODO: only training and testing on 1 fold
    break


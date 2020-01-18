import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler
import numpy as np
import pickle
import argparse
import random
from utils import (
    dice_score, 
    MRISegConfigParser,
    save_model,
    load_data,
    train
    )
from torch.utils.data import DataLoader
from losses.dice import (
    AvgDiceLoss,
    DiceLoss,
    DiceRecon
    )
from model import btseg_bilinear
from model.btseg import BraTSSegmentation
from datasets.data_loader import BraTSDataset


"""
Usage:

python train.py --config ./config/test.cfg --model_name ./checkpoints_aug/ --gpu 3
python train.py --config ./config/test.cfg --model_name ./checkpoints_bilinear/ --gpu 3 --upsampling bilinear
python train.py --config ./config/test.cfg --model_name ./checkpoints_2branch/ --gpu 1

"""


parser = argparse.ArgumentParser(description='Train MRI segmentation model.')
parser.add_argument('--config')
#parser.add_argument('--gpu', type=str)
parser.add_argument('--upsampling', type=str, default='bilinear', choices=['bilinear', 'deconv'])
args = parser.parse_args()
config = MRISegConfigParser(args.config)

import os
if not os.path.exists('checkpoints'):
  print('[INFO] Make dir %s' % 'checkpoints')
  os.mkdir('checkpoints')
checkpoints_dir = 'checkpoints/' + config.model_name + "/"
if not os.path.exists('checkpoints/' + config.model_name):
  print('[INFO] Make dir %s' % 'checkpoints/' + config.model_name)
  os.mkdir(checkpoints_dir)

if config.deterministic_train:
  seed = 0
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

# Vision FAQ explicitly asks not to do this.
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

brats_data = BraTSDataset(config.data_dir, config.labels, modes=config.modes)

trainloader, testloader = load_data(brats_data)

input_channels = len(config.modes)
output_channels = len(config.labels)
#output_channels = len(config.labels) + 1

if args.upsampling == 'bilinear':
  model = btseg_bilinear.BraTSSegmentation(input_channels, output_channels)
elif args.upsampling == 'deconv':
  model = BraTSSegmentation(input_channels, output_channels)
else:
  raise('ERROR')

model = model.to(device)
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1)
#optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
# model_name = config.model_name
optimizer = \
    optim.Adam(model.parameters(), lr=1e-4, weight_decay=config.weight_decay)
train(model, AvgDiceLoss(), optimizer, trainloader, testloader, 
#train(model, DiceRecon(), optimizer, trainloader, testloader, 
    config.max_epochs, device, name=config.model_name, checkpoint_dir=checkpoints_dir)


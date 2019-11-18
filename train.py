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
from losses.dice import DiceLoss
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
parser.add_argument('--gpu', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--upsampling', type=str, default='deconv', choices=['bilinear', 'deconv'])
args = parser.parse_args()
config = MRISegConfigParser(args.config)
from pprint import pprint
pprint(vars(args))

import os
if not os.path.exists(args.model_name):
    print('[INFO] Make dir %s' % args.model_name)
    os.mkdir(args.model_name)

if config.deterministic_train:
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# from multiprocessing import set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

brats_data = BraTSDataset(config.data_dir, config.labels, modes=config.modes)

trainloader, testloader = load_data(brats_data)


input_channels = len(config.modes)
output_channels = len(config.labels)

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
train(model, DiceLoss(), optimizer, trainloader, testloader, 
        config.max_epochs, device, name=args.model_name)


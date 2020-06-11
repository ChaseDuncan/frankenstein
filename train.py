import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
import argparse
import random
from utils import (
    MRISegConfigParser,
    save_model,
    load_data,
    train,
    validate,
    create_dir
    )

from torch.utils.data import DataLoader
from factory.scheduler import PolynomialLR
from losses import losses
from model import vaereg
from datasets.data_loader import BraTSDataset

"""
Usage:

python train.py --config ./config/test.cfg --model_name ./checkpoints_aug/ --gpu 3
python train.py --config ./config/test.cfg --model_name ./checkpoints_bilinear/ --gpu 3 --upsampling bilinear
python train.py --config ./config/test.cfg --model_name ./checkpoints_2branch/ --gpu 1

"""


parser = argparse.ArgumentParser(description='Train MRI segmentation model.')
parser.add_argument('--config')
parser.add_argument('--upsampling', type=str, default='bilinear', choices=['bilinear', 'deconv'])
args = parser.parse_args()
config = MRISegConfigParser(args.config)

device = torch.device('cuda')

for d in ['checkpoints', config.log_dir]:
  create_dir(d, config.model_name)

if config.deterministic_train:
  seed = 0
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

# TODO: just pass config and figure it out
#brats_data = BraTSDataset(config.data_dir, config.labels, modes=config.modes, debug=True)
#train, test = torch.utils.data.random_split(brats_data, [8, 2]) 
#trainp = 228
#testp = 57
trainp = 285
testp = 0

brats_data = BraTSDataset(config.data_dir, modes=config.modes, debug=config.debug, dims=config.dims)
#train_split, test_split = torch.utils.data.random_split(brats_data, [trainp, testp]) 
#
#trainloader = DataLoader(train_split, batch_size=1, shuffle=True, num_workers=0)
#testloader = DataLoader(test_split, batch_size=1, shuffle=True, num_workers=0)
trainloader = DataLoader(brats_data, batch_size=6, shuffle=True, num_workers=0)
testloader = None

# TODO: Replace with builder.
if config.model_type == 'baseline':
  model = vaereg.UNet()
  model = nn.DataParallel(model)
  model = model.to(device)
if config.model_type == 'reconreg':
  model = vaereg.ReconReg()
  model = model.to(device)
if config.model_type == 'vaereg':
  model = vaereg.VAEreg()

# TODO: optimizer factory
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1)
#optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
# model_name = config.model_name
optimizer = \
    optim.Adam(model.parameters(), lr=1e-4, weight_decay=config.weight_decay)

writer = SummaryWriter(log_dir=config.log_dir+config.model_name+'/')
scheduler = PolynomialLR(optimizer, config.epochs)
loss = losses.build(config)

for epoch in range(1, config.epochs):
  train(model, loss, optimizer, trainloader, device)
  
  #Only validate every x epochs
  if epoch % 5 != 0:
    scheduler.step()
    continue

  train_dice, train_dice_agg, train_loss, test_dice, test_dice_agg, test_loss =\
      validate(model, loss, trainloader, testloader, device)

  # Log validation
  writer.add_scalar('Loss/train', train_loss, epoch)
  writer.add_scalar('Dice/train/ncr&net', train_dice[0], epoch)
  writer.add_scalar('Dice/train/ed', train_dice[1], epoch)
  writer.add_scalar('Dice/train/et', train_dice[2], epoch)
  writer.add_scalar('Dice/train/et_agg', train_dice_agg[0], epoch)
  writer.add_scalar('Dice/train/wt_agg', train_dice_agg[1], epoch)
  writer.add_scalar('Dice/train/tc_agg', train_dice_agg[2], epoch)

  if test_dice and test_loss:
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Dice/test/ncr&net', test_dice[0], epoch)
    writer.add_scalar('Dice/test/ed', test_dice[1], epoch)
    writer.add_scalar('Dice/test/et', test_dice[2], epoch)
    # TODO: make this just test and add agg score
    print("epoch: {}\ttrain loss: {}\ttrain dice: {}\t\
        test loss: {}\t test dice: {}".format(epoch, train_loss, 
        [ d.item() for d in train_dice ], test_loss, [ d.item() for d in test_dice ])) 
  print("epoch: {} ||| train loss: {} ||| train dice: {} ||| train dice agg: {}".format(epoch, 
    train_loss, [ d.item() for d in train_dice ], [ d.item() for d in train_dice_agg ])) 
  save_model(config.model_name, epoch, writer, model, optimizer) 
  scheduler.step()
  

import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler
import numpy as np
from configparser import SafeConfigParser
import pickle
from tqdm import tqdm
from utils import dice_score
from torch.utils.data import DataLoader
from losses.dice import DiceLoss
from model.btseg import BraTSSegmentation
from datasets.data_loader import BraTSDataset

config = SafeConfigParser()
config.read("config/test.cfg")

train_split = config.getfloat('train_params', 'train_split')
data_dir = config.get('data', 'data_dir')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

brats_data = BraTSDataset(data_dir)
num_examples = len(brats_data)
data_indices = np.arange(num_examples)
deterministic_test = True
# Fix stochasticity in data sampling
if deterministic_test:
    np.random.seed(0)

# TODO: Doesn't really seem to belong here. Make a new
# class for handling this or push it to the dataloader?
np.random.shuffle(data_indices)
split_idx = int(num_examples*train_split)
test_sampler = sampler.SubsetRandomSampler(data_indices[split_idx:])
testloader = DataLoader(brats_data, 
        batch_size=1, sampler=test_sampler)

model = BraTSSegmentation(input_channels=2) 

# TODO: continue training from checkpoint
checkpoint = torch.load('checkpoints/best_test')
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)

model.eval()
sum_test_dice = 0.0
with torch.no_grad():
    for test_ex in tqdm(testloader):
        test_src, test_target = test_ex
        test_src = test_src.to(device, dtype=torch.float)
        test_target = test_target.to(device, dtype=torch.float)
        test_output = model(test_src)
        sum_test_dice += dice_score(test_output, test_target).cpu()

avg_eval_dice_by_class = sum_test_dice / len(testloader)

print("Average eval Dice: {}".format(avg_eval_dice_by_class))


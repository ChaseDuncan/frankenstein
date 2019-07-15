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

deterministic_train = config.getboolean('train_params', 'deterministic_train')
train_split = config.getfloat('train_params', 'train_split')
data_dir = config.get('data', 'data_dir')
model_name = config.get('meta', 'model_name')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

brats_data = BraTSDataset(data_dir)
num_examples = len(brats_data)
data_indices = np.arange(num_examples)

# Fix stochasticity in data sampling
if deterministic_train:
    np.random.seed(0)

# TODO: Doesn't really seem to belong here. Make a new
# class for handling this or push it to the dataloader?
np.random.shuffle(data_indices)
split_idx = int(num_examples*train_split)
train_sampler = sampler.SubsetRandomSampler(data_indices[:split_idx])
test_sampler = sampler.SubsetRandomSampler(data_indices[split_idx:])
trainloader = DataLoader(brats_data, 
        batch_size=1, sampler=train_sampler)
testloader = DataLoader(brats_data, 
        batch_size=1, sampler=test_sampler)

# Fix stochasticity in model params, etc.
if deterministic_train:
    torch.manual_seed(0)

model = BraTSSegmentation(input_channels=2) 

# TODO: continue training from checkpoint
checkpoint = torch.load('checkpoints/test')
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
loss = DiceLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
best_loss = 1.0

epoch = 0
best_eval = 0.0
# TODO: restart training from checkpoint.
epoch = checkpoint['epoch']
best_eval = 0.26033

# losses = {} # TODO: checkpoint issue
losses = pickle.load(open("losses.pkl", "rb"))
while(True):
    epoch+=1
    total_loss = 0.0
    losses[epoch] = []
    model.train()
    
    for train_ex in tqdm(trainloader):
        optimizer.zero_grad()
        src, target = train_ex
        src = src.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        output = model(src)
        cur_loss = loss(output, target)
        total_loss+=cur_loss
        losses[epoch].append(cur_loss)
        cur_loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / num_examples

    sum_test_dice = torch.zeros(3)
    
    pickle.dump(losses, open("losses.pkl", "wb"))
    model.eval()

    with torch.no_grad():
        for test_ex in tqdm(testloader):
            test_src, test_target = test_ex
            test_src = test_src.to(device, dtype=torch.float)
            test_target = test_target.to(device, dtype=torch.float)
            test_output = model(test_src)
            sum_test_dice += dice_score(test_output, test_target).cpu()

    avg_eval_dice_by_class = sum_test_dice / len(testloader)

    print("Saving model after training epoch {}. Average train loss: {} \
            Average eval Dice: {}".format(epoch, avg_train_loss, 
                avg_eval_dice_by_class))

    torch.save({'epoch': epoch, 
        'loss': avg_train_loss, 
        'eval': avg_eval_dice_by_class,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        'checkpoints/'+ model_name)

    avg_eval_dice = torch.sum(avg_eval_dice_by_class) / 3
    if avg_eval_dice > best_eval:
        torch.save({'epoch': epoch, 
            'loss': avg_train_loss, 
            'eval': avg_eval_dice,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            'checkpoints/'+ 'best_' +model_name)
                
    best_eval = avg_eval_dice

    if best_eval > 0.80 or epoch > 999: # TODO: better stopping criteria. Convergence threshold?
        break

print("Training complete.")


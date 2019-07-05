import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from losses.dice import DiceLoss
from model.btseg import BraTSSegmentation
from datasets.data_loader import BraTSDataset

deterministic_train = True
data_dir = "/home/chase/data/miccai-brats-2018-data-training/"
model_name = 't1_t1ce_0.pt'
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
brats_data = BraTSDataset(data_dir)
num_examples = len(brats_data)
data_indices = np.arange(num_examples)

if deterministic_train:
    np.random.seed(0)

np.random.shuffle(data_indices)
split_idx = int(num_examples*0.8)

train_sampler = sampler.SubsetRandomSampler(data_indices[:split_idx])
test_sampler = sampler.SubsetRandomSampler(data_indices[split_idx:])
trainloader = DataLoader(brats_data, 
        batch_size=1, sampler=train_sampler)
testloader = DataLoader(brats_data, 
        batch_size=1, sampler=test_sampler)

if deterministic_train:
    torch.manual_seed(0)
model = BraTSSegmentation(input_channels=2) 

#checkpoint = torch.load('checkpoints/zero.pt')
#model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
loss = DiceLoss()
#optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
debug = False
best_loss = 1.0

def dice_score(preds, targets):
    num_vec = 2*torch.einsum('cijk, cijk ->c', \
            [preds.squeeze(), targets.squeeze()])
    denom = torch.einsum('cijk, cijk -> c', \
            [preds.squeeze(), preds.squeeze()]) +\
                torch.einsum('cijk, cijk -> c', \
                [targets.squeeze(), targets.squeeze()])
    avg_dice = torch.sum(num_vec / denom) / 3.0
    return avg_dice

epoch = 0
best_eval = 0.0

while(True):
    epoch+=1
    total_loss = 0.0

    model.train()
    for train_ex in tqdm(trainloader):
        optimizer.zero_grad()
        src, target = train_ex
        src = src.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        output = model(src)
        cur_loss = loss(output, target)
        total_loss+=cur_loss
        cur_loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / num_examples

    sum_test_dice = torch.zeros(3)
     
    model.eval()
    with torch.no_grad():
        for test_ex in tqdm(testloader):
            test_src, test_target = test_ex
            test_src = test_src.to(device, dtype=torch.float)
            test_target = target.to(device, dtype=torch.float)
            test_output = model(test_src)
            sum_test_dice += dice_score(test_output, test_target)
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

    if best_eval > 0.85:
        break

print("Training complete.")


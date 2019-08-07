import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler
import numpy as np
import torchsummary
import pickle
import argparse

from tqdm import tqdm
from utils import (
        dice_score, 
        MRISegConfigParser,
        save_model
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
def CV(dataset, batch_size=1, k = 5, deterministic_train=False):
    num_examples = len(brats_data)
    data_indices = np.arange(num_examples)
    if config.deterministic_train:
        np.random.seed(random_seed)
    np.random.shuffle(data_indices)
    folds = np.array(np.split(data_indices, k))

    cv_trainloader = []
    cv_testloader = []

    for i in range(len(folds)):
        mask = np.zeros(len(folds), dtype=bool)
        mask[i] = True
        train_folds = np.hstack(folds[~mask])
        test_fold = folds[mask][0]
        cv_trainloader.append(DataLoader(dataset, 
            batch_size, num_workers=8, sampler=sampler.SubsetRandomSampler(train_folds)))
        cv_testloader.append(DataLoader(dataset, 
            batch_size, num_workers=8, sampler=sampler.SubsetRandomSampler(test_fold)))
    return cv_trainloader, cv_testloader


def train(model, optimizer, train_data_loader, test_data_loader, max_epoch, 
        name="default", best_eval=0.0, epoch=0):
    avg_train_losses = []

    while(True):
        epoch+=1
        total_loss = 0.0
        model.train()
        
        for train_ex in tqdm(trainloader):
            optimizer.zero_grad()
            src, target = train_ex
            #torchsummary.summary(model, input_size=(src.shape[1], src.shape[2], src.shape[3], src.shape[4]))
            src = src.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            output = model(src)
            cur_loss = loss(output, target)
            total_loss+=cur_loss
            cur_loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(trainloader)
        avg_train_losses.append(avg_train_loss)

        sum_test_dice = 0.0

        with torch.no_grad():
            model.eval()
            for test_ex in tqdm(testloader):
                test_src, test_target = test_ex
                test_src = test_src.to(device, dtype=torch.float)
                test_target = test_target.to(device, dtype=torch.float)
                test_output = model(test_src)
                sum_test_dice += dice_score(test_output, test_target).cpu()

        eval_dice = sum_test_dice / len(testloader)

        print("Saving model after training epoch {}. Average train loss: {} \
                            Average eval Dice: {}".format(epoch, avg_train_loss, 
                                                eval_dice))
        save_model(name, epoch, avg_train_losses, eval_dice, model, optimizer)

        avg_eval_dice = torch.sum(eval_dice) / 3
        if avg_eval_dice > best_eval:
            save_model(name, epoch, avg_train_losses, eval_dice, model, optimizer)
            best_dice_by_class = eval_dice

        best_eval = avg_eval_dice
        if epoch > max_epoch: # TODO: better stopping criteria. Convergence threshold?
            break

    print("Training complete.")
    return best_dice_by_class

cv_trainloaders, cv_testloaders = CV(brats_data, batch_size=1, deterministic_train=config.deterministic_train)

# Fix stochasticity in model params, etc.

for i, (trainloader, testloader) in enumerate(zip(cv_trainloaders, cv_testloaders)):
    if config.deterministic_train:
        torch.manual_seed(random_seed)

    input_channels = len(config.modes)
    output_channels = len(config.labels)
    #output_channels = 3
    model = BraTSSegmentation(input_channels, output_channels) 
    model = model.to(device)
    loss = DiceLoss()

    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1)
    #optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
    model_name = "{}_fold_{}".format(config.model_name, i)
    optimizer = \
            optim.Adam(model.parameters(), lr=1e-4, weight_decay=config.weight_decay)
    train(model, optimizer, trainloader, testloader, config.max_epochs,
        name=model_name, best_eval=0.0, epoch=0)

    # TODO: only training and testing on 1 fold
    break



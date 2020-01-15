import torch
import numpy as np
import json

from configparser import ConfigParser
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler
from tqdm import tqdm

def dice_score(preds, targets):
  num_vec = 2*torch.einsum('cijk, cijk ->c', \
      [preds.squeeze(0), targets.squeeze(0)])
  denom = torch.einsum('cijk, cijk -> c', \
      [preds.squeeze(0), preds.squeeze(0)]) +\
      torch.einsum('cijk, cijk -> c', \
      [targets.squeeze(0), targets.squeeze(0)])
  dice = num_vec / denom
  return dice

class MRISegConfigParser():
  def __init__(self, config_file):
    config = ConfigParser()
    config.read(config_file)

    self.deterministic_train = \
        config.getboolean('train_params', 'deterministic_train')
    self.train_split = config.getfloat('train_params', 'train_split')
    self.weight_decay = config.getfloat('train_params', 'weight_decay')
    self.max_epochs = config.getint('train_params', 'max_epochs')
    self.data_dir = config.get('data', 'data_dir')
    self.model_name = config.get('meta', 'model_name')
    self.modes = json.loads(config.get('data', 'modes'))
    self.labels = json.loads(config.get('data', 'labels'))


# TODO: clean this up vis a vis checkpoints vs saving model, etc.
def save_model(name, epoch, avg_train_losses, eval_dice, model, optimizer):
  torch.save({'epoch': epoch,
    'losses': avg_train_losses,
    'eval': eval_dice,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()},
    name)

def load_data(dataset):
  cv_trainloader, cv_testloader = cross_validation(dataset)
  return cv_trainloader[0], cv_testloader[0]

def cross_validation(dataset, batch_size=1, k = 5):
  num_examples = len(dataset)
  data_indices = np.arange(num_examples)
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
      batch_size, num_workers=0, sampler=sampler.SubsetRandomSampler(train_folds)))
    cv_testloader.append(DataLoader(dataset,
      batch_size, num_workers=0, sampler=sampler.SubsetRandomSampler(test_fold)))
    return cv_trainloader, cv_testloader


def train(model, loss, optimizer, train_data_loader, test_data_loader, max_epoch, device,
    name="default", best_eval=0.0, epoch=0, checkpoint_dir="./"):

  avg_train_losses = []

  while(True):
    epoch+=1
    total_loss = 0.0
    model.train()

    for train_ex in tqdm(train_data_loader):
      optimizer.zero_grad()
      src, target = train_ex
      src = src.to(device, dtype=torch.float)
      target = target.to(device, dtype=torch.float)
      output = model(src)
      #output, recon, mu, logvar = model(src)

      cur_loss = loss(output, target)
      # print('cur_loss: ', cur_loss)
      total_loss += cur_loss
      cur_loss.backward()
      optimizer.step()

    avg_train_loss = total_loss / len(train_data_loader)
    avg_train_losses.append(avg_train_loss)

    sum_test_dice = 0.0

    with torch.no_grad():
      model.eval()
      for test_ex in tqdm(test_data_loader):
        test_src, test_target = test_ex
        test_src = test_src.to(device, dtype=torch.float)
        test_target = test_target.to(device, dtype=torch.float)
        test_output = model(test_src)
        sum_test_dice += dice_score(test_output, test_target).cpu()

    eval_dice = sum_test_dice / len(test_data_loader)

    print("Saving model after training epoch {} in {}. Average train loss: {} \
        Average eval Dice: {}".format(epoch, name + '_test', avg_train_loss, eval_dice))

    save_model(name, epoch, avg_train_losses, eval_dice, model, optimizer)

    avg_eval_dice = torch.sum(eval_dice) / len(eval_dice)

    if avg_eval_dice > best_eval:
      save_model(name+'_best', epoch, avg_train_losses, eval_dice, model, optimizer)
      best_dice_by_class = eval_dice

    best_eval = avg_eval_dice
    if epoch > max_epoch: # TODO: better stopping criteria. Convergence threshold?
      break

  print("Training complete.")
  return best_dice_by_class


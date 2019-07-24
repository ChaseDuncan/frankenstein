import torch
import json
from configparser import ConfigParser

def dice_score(preds, targets):
    num_vec = 2*torch.einsum('cijk, cijk ->c', \
            [preds.squeeze(), targets.squeeze()])
    denom = torch.einsum('cijk, cijk -> c', \
            [preds.squeeze(), preds.squeeze()]) +\
            torch.einsum('cijk, cijk -> c', \
            [targets.squeeze(), targets.squeeze()])
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

def save_model(name, epoch, avg_train_losses, eval_dice, model, optimizer):
        torch.save({'epoch': epoch, 
            'losses': avg_train_losses, 
            'eval': eval_dice,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            'checkpoints/'+ name)



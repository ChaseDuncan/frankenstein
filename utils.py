import torch

def dice_score(preds, targets):
    num_vec = 2*torch.einsum('cijk, cijk ->c', \
            [preds.squeeze(), targets.squeeze()])
    denom = torch.einsum('cijk, cijk -> c', \
            [preds.squeeze(), preds.squeeze()]) +\
                torch.einsum('cijk, cijk -> c', \
                [targets.squeeze(), targets.squeeze()])
    dice = num_vec / denom
    return dice

def parse_config(config_file):
    config = ConfigParser()
    config.read(config_file)

    deterministic_train = config.getboolean('train_params', 'deterministic_train')
    train_split = config.getfloat('train_params', 'train_split')
    data_dir = config.get('data', 'data_dir')
    model_name = config.get('meta', 'model_name')
    modes = json.loads(config.get('data', 'modes'))

    return config


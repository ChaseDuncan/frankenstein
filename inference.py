import torch

from model.btseg import BraTSSegmentation
from datasets.test_data_loader import BraTSDataset
import nibabel as nib

data_dir = "/home/chase/data/miccai-brats-2018-data-training/"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

testloader = BraTSDataset(data_dir)
#checkpoint = torch.load('checkpoints/zero.pt')
#checkpoint = torch.load('checkpoints/best_overfit.pt')
checkpoint = torch.load('checkpoints/best_overfit_t1_t1ce.pt')
model = BraTSSegmentation(input_channels=2) 
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

for i, data in enumerate(testloader):
    src, target = data
    src = src.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    output = model(src)
    z_mat = torch.zeros(output.shape).to(device)
    o_mat = torch.ones(output.shape).to(device)
    preds = torch.where(output>0.5, o_mat, z_mat)
    
    intersect = torch.einsum('cijk, cijk -> c', [preds.squeeze(), target.squeeze()])
    pred_cards = torch.einsum('cijk, cijk -> c', [preds.squeeze(), preds.squeeze()])
    target_cards = torch.einsum('cijk, cijk -> c', [target.squeeze(), target.squeeze()])

    dice_scores = 2*intersect / (pred_cards + target_cards)
    print(dice_scores)
    break

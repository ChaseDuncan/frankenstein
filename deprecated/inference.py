import os
import torch
import numpy as np
import argparse

from utils import MRISegConfigParser
from model.btseg import BraTSSegmentation

from torch.utils.data import DataLoader
from datasets.data_loader import BraTSDataset
import nibabel as nib

parser = argparse.ArgumentParser(description='Perform inference using trained MRI segmentation model.')
parser.add_argument('--config')
args = parser.parse_args()

config = MRISegConfigParser(args.config)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

testloader = DataLoader(BraTSDataset(config.data_dir, modes=config.modes), batch_size=1)
#checkpoint = torch.load('checkpoints/zero.pt')
#checkpoint = torch.load('checkpoints/best_overfit.pt')
checkpoint = torch.load('checkpoints/best_'+config.model_name)
model = BraTSSegmentation(input_channels=len(config.modes)) 
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

for i, data in enumerate(testloader):
    src, target = data

    src_npy = src.squeeze().numpy()[1, :, :, :]
    img = nib.Nifti1Image(src_npy, np.eye(4))
    nib.save(img, os.path.join('scratch','test.nii.gz'))

    et_npy = target.squeeze().numpy()[0, :, :, :]
    et_img = nib.Nifti1Image(et_npy, np.eye(4))
    nib.save(et_img, os.path.join('scratch','et_gt.nii.gz'))

    tc_npy = target.squeeze().numpy()[1, :, :, :]
    tc_img = nib.Nifti1Image(tc_npy, np.eye(4))
    nib.save(tc_img, os.path.join('scratch','tc_gt.nii.gz'))

    wt_npy = target.squeeze().numpy()[2, :, :, :]
    wt_img = nib.Nifti1Image(wt_npy, np.eye(4))
    nib.save(wt_img, os.path.join('scratch','wt_gt.nii.gz'))

    src = src.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    output = model(src)

    z_mat = torch.zeros(output.shape).to(device)
    o_mat = torch.ones(output.shape).to(device)
    preds = torch.where(output>0.5, o_mat, z_mat)
    
    et_pred = preds.cpu().squeeze().numpy()[0, :, :, :]
    pred_img = nib.Nifti1Image(et_pred, np.eye(4))
    nib.save(pred_img, os.path.join('scratch', 'et_pd.nii.gz'))

    tc_pred = preds.cpu().squeeze().numpy()[1, :, :, :]
    pred_img = nib.Nifti1Image(tc_pred, np.eye(4))
    nib.save(pred_img, os.path.join('scratch', 'tc_pd.nii.gz'))

    wt_pred = preds.cpu().squeeze().numpy()[2, :, :, :]
    pred_img = nib.Nifti1Image(wt_pred, np.eye(4))
    nib.save(pred_img, os.path.join('scratch', 'wt_pd.nii.gz'))
    
    intersect = torch.einsum('cijk, cijk -> c', 
            [preds.squeeze(), target.squeeze()])
    print(intersect)
    pred_cards = torch.einsum('cijk, cijk -> c', 
            [preds.squeeze(), preds.squeeze()])
    print(pred_cards)
    target_cards = torch.einsum('cijk, cijk -> c', 
            [target.squeeze(), target.squeeze()])
    print(target_cards)

    dice_scores = 2*intersect / (pred_cards + target_cards)
    print(dice_scores)
    break

import os
import torch
import numpy as np

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
    src_npy = src.squeeze().numpy()[1, :, :, :]
    wt_npy = target.squeeze().numpy()[2, :, :, :]
    img = nib.Nifti1Image(src_npy, np.eye(4))
    wt_img = nib.Nifti1Image(wt_npy, np.eye(4))
    nib.save(img, os.path.join('scratch','test.nii.gz'))
    nib.save(wt_img, os.path.join('scratch','wt_gt.nii.gz'))
    src = src.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    output = model(src)
    z_mat = torch.zeros(output.shape).to(device)
    o_mat = torch.ones(output.shape).to(device)
    preds = torch.where(output>0.5, o_mat, z_mat)
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

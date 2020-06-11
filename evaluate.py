import os
import torch
import torch.nn as nn
from model import vaereg
from tqdm import tqdm
import numpy as np

import argparse
from torch.utils.data import DataLoader
from data_loader import BraTSDataset
from utils import MRISegConfigParser
import os
import nibabel as nib

device = torch.device('cuda')
model = vaereg.UNet()

parser = argparse.ArgumentParser(description='Train MRI segmentation model. Provide config file for model, --config, and path to evaluation data directory, --data.')
parser.add_argument('--data')
args = parser.parse_args()

#checkpoint_file='/shared/mrfil-data/cddunca2/gliomaseg/baseline/checkpoints/checkpoint-300.pt'
#checkpoint_file = 'checkpoints/vaereg-fulltrain-vision/vaereg-fulltrain'
#checkpoint_file = 'checkpoints/vaereg-fulltrain/vaereg-fulltrain'
#checkpoint_file = 'cddunca2/brats2020/debug/checkpoints/checkpoint-100.pt'
#checkpoint_file = 'cddunca2/brats2020/debug/checkpoints/checkpoint-75.pt'
#checkpoint_file = 'cddunca2/brats2020/debug/checkpoints/checkpoint-10.pt'
checkpoint_file = 'checkpoints/baseline-vision/baseline'
annotations_dir = 'cddunca2/brats2020/debug/annotations/baseline-vision/'
#annotations_dir = '/shared/mrfil-data/cddunca2/OSFData/NCM0014-segmentation/'
os.makedirs(annotations_dir, exist_ok=True)

checkpoint = torch.load(checkpoint_file)
# Name of state dict in vision checkpoint
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#l = [module for module in model.modules() if type(module) != nn.Sequential]
#model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(device)

brats_data = BraTSDataset('/dev/shm/brats2018validation/', dims=[160, 192, 128])
#brats_data = BraTSDataset('/dev/shm/brats2018/', dims=[160, 192, 128])
#brats_data = BraTSDataset('/shared/mrfil-data/cddunca2/OSFData/NCM0014/', 
#                      dims=[160, 192, 128])
dataloader = DataLoader(brats_data, batch_size=1, num_workers=0)

with torch.no_grad():
  model.eval()
  dims=[160, 192, 128]
  for src, tgt in tqdm(dataloader):
    ID = tgt[0].split("/")[-1]
    #ID = 'debug.nii.gz'
    # This is ugly, loading in the image just to get its dimensions for uncropping
    src = src.to(device, dtype=torch.float)
    output = model(src)
    x_off = int((182 - dims[0]) / 2)
    y_off = int((218 - dims[1]) / 2)
    z_off = int((182 - dims[2]) / 2)
    m = nn.ConstantPad3d((z_off, z_off, y_off, y_off, x_off, x_off), 0)
    ncr_net = m(output[0, 0, :, :, :])
    ed = m(output[0, 1, :, :, :])
    et = m(output[0, 2, :, :, :])

    label = torch.zeros((182, 218, 182))
    label[torch.where(ncr_net > 0.5)] = 1
    #img = nib.Nifti1Image(label.numpy(), aff)
    label[torch.where(ed > 0.5)] = 2
    label[torch.where(et > 0.5)] = 4
    #aff = np.array([[-1, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])
    #ncr_net = output[0, 0, :, :, :]
    #ncr_net = ncr_net[torch.where(ncr_net > 0.5)] = 1
    #img = nib.Nifti1Image(ncr_net.cpu().numpy(), np.eye(4))
    #img = nib.Nifti1Image(label.numpy(), aff)
    
    img = nib.Nifti1Image(label.numpy(), np.eye(4))
    img.to_filename(os.path.join(annotations_dir, ID))
    #tgt0 = nib.Nifti1Image(tgt.squeeze()[0, :, :, :].numpy(), np.eye(4))
    #tgt0.to_filename(os.path.join(annotations_dir, 'ground_truth_0.nii.gz'))
    #tgt1 = nib.Nifti1Image(tgt.squeeze()[1, :, :, :].numpy(), np.eye(4))
    #tgt1.to_filename(os.path.join(annotations_dir, 'ground_truth_1.nii.gz'))
    #tgt2 = nib.Nifti1Image(tgt.squeeze()[2, :, :, :].numpy(), np.eye(4))
    #tgt2.to_filename(os.path.join(annotations_dir, 'ground_truth_2.nii.gz'))
    break



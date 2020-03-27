import os
import torch
import torch.nn as nn
from model import vaereg
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from datasets.data_loader import BraTSDataset

import nibabel as nib

device = torch.device('cuda')
model = vaereg.UNet()
#checkpoint = \
#    torch.load('checkpoints/vaereg-fulltrain-smallcrop-eloss/vaereg-fulltrain-smallcrop-eloss', 
#map_location='cuda:0')
checkpoint = \
    torch.load('checkpoints/vaereg-fulltrain-smallcrop-eloss/vaereg-fulltrain-smallcrop-eloss', 
        map_location='cuda:0')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)

brats_data = \
    BraTSDataset('/data/cddunca2/brats2018/validation/', dims=[128, 128, 128])
dataloader = DataLoader(brats_data, batch_size=1, num_workers=0)
dims=[128, 128, 128]
with torch.no_grad():
  model.eval()
  for src, tgt in tqdm(dataloader):
    ID = tgt[0].split("/")[5]
    src = src.to(device, dtype=torch.float)
    
    output = model(src)
    x_off = int((240 - dims[0]) / 4)*2
    y_off = int((240 - dims[1]) / 4)*2
    m = nn.ConstantPad3d((13, 14, x_off, x_off, y_off, y_off), 0)
    
    ncr_net = m(output[0, 0, :, :, :])
    ed = m(output[0, 1, :, :, :])
    et = m(output[0, 2, :, :, :])
    
    label = torch.zeros((240, 240, 155))
    label[torch.where(ncr_net > 0.5)] = 1
    label[torch.where(ed > 0.5)] = 2
    label[torch.where(et > 0.5)] = 4

    img = nib.Nifti1Image(label.numpy(), np.eye(4))
     
    img.to_filename(os.path.join('annotations', ID+'.nii.gz'))

       


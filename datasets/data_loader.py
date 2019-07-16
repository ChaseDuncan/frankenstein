import os

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, data_dir, modes=['t1', 't1ce', 't2', 'flair'], transform=lambda x: x):
        # store filenames. expects data_dir/{HGG, LGG}/
        # TODO: should HGG and LGG be separated?
        self.filenames = \
                [ data_dir + "/HGG/" + f + "/" for f in os.listdir(data_dir + "/HGG/") ]
        self.filenames.extend([ data_dir + "/LGG/" + f + "/"\
                for f in os.listdir(data_dir + "/LGG/") ])
        self.filenames = [ f + d for f in self.filenames for d in os.listdir(f) ]
        
        self.t1 = sorted([ f for f in self.filenames if "t1.nii.gz" in f ])
        self.t1ce = sorted([ f for f in self.filenames if "t1ce.nii.gz" in f ])
        self.t2 = sorted([ f for f in self.filenames if "t2.nii.gz" in f ])
        self.flair = sorted([ f for f in self.filenames if "flair.nii.gz" in f ])
        self.segs = sorted([ f for f in self.filenames if "seg.nii.gz" in f ])
        self.modes = modes
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.t1)

    def __getitem__(self, idx):
        data = []
        # open image and apply transform if applicable
        if 't1' in self.modes: 
            data.append(self.transform(nib.load(self.t1[idx]).get_fdata()))
        if 't1ce' in self.modes:
            data.append(self.transform(nib.load(self.t1ce[idx]).get_fdata()))
        if 't2' in self.modes:
            data.append(self.transform(nib.load(self.t2[idx]).get_fdata()))
        if 'flair' in self.modes:
            data.append(self.transform(nib.load(self.flair[idx]).get_fdata()))

        seg = nib.load(self.segs[idx]).get_fdata()

        # TODO: move this out
        #img_t1 = img_t1[56:-56, 56:-56, 14:-13]  
        #img_t1ce = img_t1ce[56:-56, 56:-56, 14:-13]  
        #img_t2 = img_t2[56:-56, 56:-56, 14:-13]  
        #img_flair = img_flair[56:-56, 56:-56, 14:-13]  
        data = [d[56:-56, 56:-56, 14:-13] for d in data] 
        seg = seg[56:-56, 56:-56, 14:-13]  

        seg_et = np.zeros(seg.shape)
        seg_et[np.where(seg==4)] = 1
        seg_tc = np.zeros(seg.shape)
        seg_tc[np.where(seg==1) or np.where(seg==4)] = 1
        seg_wt = np.zeros(seg.shape)
        seg_wt[np.where(seg>0)] = 1

        src = np.stack(data)
        target = np.stack((seg_et, seg_tc, seg_wt))

        return torch.from_numpy(src), torch.from_numpy(target)



import os

import numpy as np
import nibabel as nib
import torch
from torchvision import transforms
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

        # TODO: move cropping out
        if 't1' in self.modes: 
            t1 = self.transform(nib.load(self.t1[idx]).get_fdata())
            t1 = t1[56:-56, 56:-56, 14:-13]
            t1 = torch.from_numpy(t1)
            t1_mean = torch.mean(t1)
            t1_std = torch.std(t1)
            t1_trans = transforms.Normalize(t1, t1_mean, t1_std)
            data.append(t1)

        if 't1ce' in self.modes:
            t1ce = self.transform(nib.load(self.t1ce[idx]).get_fdata())
            t1ce = t1ce[56:-56, 56:-56, 14:-13]
            t1ce = torch.from_numpy(t1ce)
            t1ce_mean = torch.mean(t1ce)
            t1ce_std = torch.std(t1ce)
            t1ce_trans = transforms.Normalize(t1ce, t1ce_mean, t1ce_std)

            data.append(t1ce)

        if 't2' in self.modes:
            t2 = self.transform(nib.load(self.t2[idx]).get_fdata())
            t2 = t2[56:-56, 56:-56, 14:-13]
            t2 = torch.from_numpy(t2)
            t2_mean = torch.mean(t2)
            t2_std = torch.std(t2)
            t2_trans = transforms.Normalize(t2, t2_mean, t2_std)

            data.append(t2_trans)

        if 'flair' in self.modes:
            flair = self.transform(nib.load(self.flair[idx]).get_fdata())
            flair = flair[56:-56, 56:-56, 14:-13]
            flair = torch.from_numpy(flair)
            flair_mean = torch.mean(flair)
            flair_std = torch.std(flair)
            flair_trans = transforms.Normalize(flair, flair_mean, flair_std)

            data.append(flair_trans)

        seg = nib.load(self.segs[idx]).get_fdata()

        seg = seg[56:-56, 56:-56, 14:-13]  

        seg_et = np.zeros(seg.shape)
        seg_et[np.where(seg==4)] = 1
        seg_tc = np.zeros(seg.shape)
        seg_tc[np.where(seg==1) or np.where(seg==4)] = 1
        seg_wt = np.zeros(seg.shape)
        seg_wt[np.where(seg>0)] = 1

        src = torch.stack(data)
        target = np.stack((seg_et, seg_tc, seg_wt))

        return src, torch.from_numpy(target)



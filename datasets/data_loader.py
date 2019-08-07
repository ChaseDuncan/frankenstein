import os

import numpy as np
import nibabel as nib
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, data_dir, labels, modes=['t1', 't1ce', 't2', 'flair'], transform=lambda x: x):
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
        self.labels = labels
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
            means = [t1_mean]*t1.shape[0]
            t1_std = torch.std(t1)
            stds = [t1_std]*t1.shape[0]
            t1_trans = TF.normalize(t1, means, stds)

            data.append(t1_trans)

        if 't1ce' in self.modes:
            t1ce = self.transform(nib.load(self.t1ce[idx]).get_fdata())
            t1ce = t1ce[56:-56, 56:-56, 14:-13]
            t1ce = torch.from_numpy(t1ce)
            t1ce_mean = torch.mean(t1ce)
            means = [t1ce_mean]*t1ce.shape[0]
            t1ce_std = torch.std(t1ce)
            stds = [t1ce_std]*t1ce.shape[0]
            t1ce_trans = TF.normalize(t1ce, means, stds)

            data.append(t1ce_trans)

        if 't2' in self.modes:
            t2 = self.transform(nib.load(self.t2[idx]).get_fdata())
            t2 = t2[56:-56, 56:-56, 14:-13]
            t2 = torch.from_numpy(t2)
            t2_mean = torch.mean(t2)
            means = [t2_mean]*t2.shape[0]
            t2_std = torch.std(t2)
            stds = [t2_std]*t2.shape[0]
            t2_trans = TF.normalize(t2, means, stds)

            data.append(t2_trans)

        if 'flair' in self.modes:
            flair = self.transform(nib.load(self.flair[idx]).get_fdata())
            flair = flair[56:-56, 56:-56, 14:-13]
            flair = torch.from_numpy(flair)
            flair_mean = torch.mean(flair)
            means = [flair_mean]*flair.shape[0]
            flair_std = torch.std(flair)
            stds = [flair_std]*flair.shape[0]
            flair_trans = TF.normalize(flair, means, stds)

            data.append(flair_trans)

        seg = nib.load(self.segs[idx]).get_fdata()

        seg = seg[56:-56, 56:-56, 14:-13]  
        segs = []
        if "enhancing_tumor" in self.labels:
            seg_et = np.zeros(seg.shape)
            seg_et[np.where(seg==4)] = 1
            segs.append(seg_et)
        if "tumor_core" in self.labels:
            seg_tc = np.zeros(seg.shape)
            seg_tc[np.where(seg==1) or np.where(seg==4)] = 1
            segs.append(seg_tc)
        if "whole_tumor" in self.labels:
            seg_wt = np.zeros(seg.shape)
            seg_wt[np.where(seg>0)] = 1
            segs.append(seg_wt)

        src = torch.stack(data)
        target = np.stack(segs)
        return src, torch.from_numpy(target)



import os

import numpy as np
import nibabel as nib
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, data_dir, labels, modes=['t1', 't1ce', 't2', 'flair']):
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

    def __len__(self):
        # return size of dataset
        return len(self.t1)


    def data_aug(self, brain):
        shift_brain = brain + torch.Tensor(np.random.uniform(-0.1, 0.1, brain.shape)).double().cuda()
        scale_brain = shift_brain*torch.Tensor(np.random.uniform(0.9, 1.1, brain.shape)).double().cuda()
        return scale_brain


    # TODO: mask brain
    def std_normalize(self, d):
      ''' Subtract mean and divide by standard deviation of the image.'''
      d = torch.from_numpy(d)
      d_mean = torch.mean(d)
      means = [d_mean]*d.shape[0]
      d_std = torch.std(d)
      stds = [d_std]*d.shape[0]
      d_trans = TF.normalize(d, means, stds).cuda()
      return d_trans


    def min_max_normalization(self, d):
        return (d - d.min()) / (d.max() - d.min())


    def __getitem__(self, idx):
        data = []
        # TODO: move cropping out
        a = np.random.rand(1)

        # randomly flip along axis
        if a > 0.5:
            axis = np.random.choice([0, 1, 2], 1)[0]
        # TODO: fix code smell.
        if 't1' in self.modes:
            t1 = nib.load(self.t1[idx]).get_fdata()
            t1 = t1[56:-56, 56:-56, 14:-13]
            if a > 0.5:
                t1 = np.flip(t1, axis).copy()

            #t1_trans = self.std_normalize(t1)
            t1_trans = self.min_max_normalize(t1)
            aug_brain = self.data_aug(t1_trans)
            data.append(aug_brain)

        if 't1ce' in self.modes:
            t1ce = nib.load(self.t1ce[idx]).get_fdata()
            t1ce = t1ce[56:-56, 56:-56, 14:-13]
            if a > 0.5:
                t1ce = np.flip(t1ce, axis).copy()

            #t1ce_trans = self.std_normalize(t1ce)
            t1ce_trans = self.min_max_normalize(t1ce)
            aug_brain = self.data_aug(t1ce_trans)
            data.append(aug_brain)

        if 't2' in self.modes:
            t2 = nib.load(self.t2[idx]).get_fdata()
            t2 = t2[56:-56, 56:-56, 14:-13]
            if a > 0.5:
                t2 = np.flip(t2, axis).copy()

            #t2_trans = self.std_normalize(t2)
            t2_trans = self.min_max_normalize(t2)

            aug_brain = self.data_aug(t2_trans)
            data.append(aug_brain)

        if 'flair' in self.modes:
            flair = nib.load(self.flair[idx]).get_fdata()
            flair = flair[56:-56, 56:-56, 14:-13]
            if a > 0.5:
                flair = np.flip(flair, axis).copy()

            #flair_trans = self.std_normalize(flair)
            flair_trans = self.min_max_normalize(flair)

            aug_brain = self.data_aug(flair_trans)
            data.append(aug_brain)

        seg = nib.load(self.segs[idx]).get_fdata()

        seg = seg[56:-56, 56:-56, 14:-13]
        if a > 0.5:
            seg = np.flip(seg, axis)

        segs = []
        # See part E. of the BraTS reference
        # https://ieeexplore.ieee.org/document/6975210
        # for how these labels are derived from the annotations.
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
        if "enhancing_core" in self.labels:
            seg_tc = np.zeros(seg.shape)
            seg_tc[np.where(seg==1)] = 1
            segs.append(seg_tc)

        src = torch.stack(data)
        target = np.stack(segs)
        return src, torch.from_numpy(target)


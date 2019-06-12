import os

import nibabel as nib
import torch
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # store filenames. expects data_dir/{HGG, LGG}/
        # TODO: should HGG and LGG be separated?
        self.filenames = \
                [ data_dir + "/HGG/" + f + "/" for f in os.listdir(data_dir + "/HGG/") ]
        self.filenames.extend([ data_dir + "/LGG/" + f + "/"\
                for f in os.listdir(data_dir + "/LGG/") ])
        self.filenames = [ f + d for f in self.filenames for d in os.listdir(f) ]
        # only take t1 files
        self.input = sorted([ f for f in self.filenames if "t1.nii.gz" in f ])
        self.segs = sorted([ f for f in self.filenames if "seg.nii.gz" in f ])
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.input)

    def __getitem__(self, idx):
        # open image and apply transform if applicable
        img = nib.load(self.input[0]).get_fdata()
        seg = nib.load(self.segs[0]).get_fdata()

        # TODO: move this out
        img = img[56:-56, 56:-56, 14:-13]  
        seg = seg[56:-56, 56:-56, 14:-13]  
        return torch.from_numpy(img), torch.from_numpy(seg)


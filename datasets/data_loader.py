import os

import numpy as np
import nibabel as nib
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, data_dir, modes=['t1', 't1ce', 't2', 'flair'], 
        debug=False, dims=[240, 240, 155], augment_data = False):
        # store filenames. expects data_dir/{HGG, LGG}/
        # TODO: should HGG and LGG be separated?
        self.x_off = 0
        self.y_off = 0
        self.z_off = 0

        if dims:
          self.x_off = int((240 - dims[0]) / 4)*2
          self.y_off = int((240 - dims[1]) / 4)*2
          self.z_off = int((155 - dims[2]) / 4)*2

        filenames = []
        for (dirpath, dirnames, files) in os.walk(data_dir):
          filenames += [os.path.join(dirpath, file) for file in files if '.nii.gz' in file ]

        self.modes = []
        if 't1' in modes:
          self.modes.append(sorted([ f for f in filenames if "t1.nii.gz" in f ]))
        if 't1ce' in modes:
          self.modes.append(sorted([ f for f in filenames if "t1ce.nii.gz" in f ]))
        if 't2' in modes:
          self.modes.append(sorted([ f for f in filenames if "t2.nii.gz" in f ]))
        if 'flair' in modes:
          self.modes.append(sorted([ f for f in filenames if "flair.nii.gz" in f ]))

        self.segs = sorted([ f for f in filenames if "seg.nii.gz" in f ])

        if debug:
          self.modes = []
          if 't1' in modes:
            self.modes.append(sorted([ f for f in filenames if "t1.nii.gz" in f ])[:1])
          if 't1ce' in modes:
            self.modes.append(sorted([ f for f in filenames if "t1ce.nii.gz" in f ])[:1])
          if 't2' in modes:
            self.modes.append(sorted([ f for f in filenames if "t2.nii.gz" in f ])[:1])
          if 'flair' in modes:
            self.modes.append(sorted([ f for f in filenames if "flair.nii.gz" in f ])[:1])

          self.segs = sorted([ f for f in filenames if "seg.nii.gz" in f ])[:1]

        self.augment_data = augment_data
        # randomly flip along axis
        self.axis = None
        if self.augment_data:
          if a > 0.5:
              self.axis = np.random.choice([0, 1, 2], 1)[0]


    def __len__(self):
        # return size of dataset
        return len(self.modes[0])


    def data_aug(self, brain):
        if self.axis:
            brain = np.flip(brain, self.axis).copy()
        shift_brain = brain + torch.Tensor(np.random.uniform(-0.1, 0.1, brain.shape)).double().cuda()
        scale_brain = shift_brain*torch.Tensor(np.random.uniform(0.9, 1.1, brain.shape)).double().cuda()
        return scale_brain


    # TODO: mask brain
    # changing data type in the function is stupid
    def std_normalize(self, d):
      ''' Subtract mean and divide by standard deviation of the image.'''
      d = torch.from_numpy(d)
      d_mean = torch.mean(d)
      means = [d_mean]*d.shape[0]
      d_std = torch.std(d)
      stds = [d_std]*d.shape[0]
      d_trans = TF.normalize(d, means, stds).cuda()
      return d_trans


    def _transform_data(self, d):
      t1 = nib.load(d).get_fdata()
      # TODO: z axis cropping is still hardcoded
      t1 = t1[self.x_off:240-self.x_off, self.y_off:240-self.y_off, 13:-14]
      t1_trans = self.min_max_normalize(t1)
      #t1_trans = self.std_normalize(t1)

      if self.augment_data:
        t1_trans = self.data_aug(t1_trans)
      return t1_trans


    def min_max_normalize(self, d):
      # TODO: changing data type in the function is stupid
      d = torch.from_numpy(d)
      d = (d - d.min()) / (d.max() - d.min())
      return d.cuda()


    def __getitem__(self, idx):
      data = []
      data = [self._transform_data(m[idx]) for m in self.modes]
      src = torch.stack(data)

      target = []
      if self.segs:
        seg = nib.load(self.segs[idx]).get_fdata()
        # TODO: z axis cropping is still hardcoded
        seg = seg[self.x_off:240-self.x_off, self.y_off:240-self.y_off, 13:-14]
        if self.axis:
          seg = np.flip(seg, axis)

        segs = []
        # TODO: Wrap in a loop.
        seg_ncr_net = np.zeros(seg.shape)
        seg_ncr_net[np.where(seg==1)] = 1
        segs.append(seg_ncr_net)

        seg_ed = np.zeros(seg.shape)
        seg_ed[np.where(seg==2)] = 1
        segs.append(seg_ed)

        seg_et = np.zeros(seg.shape)
        seg_et[np.where(seg==4)] = 1
        segs.append(seg_et)
        target = torch.from_numpy(np.stack(segs))
        return src, target
      
      target = self.modes[0][idx]
      return src, target

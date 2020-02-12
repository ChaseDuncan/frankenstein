


#TODO: should use datasets/data_loader.py

from nilearn.image import resample_img
import nibabel as nib
import os

data_dir = '/data/cddunca2/brats2018/'
scaling = 1.3

filenames = \
   [ data_dir + "/HGG/" + f + "/" for f in os.listdir(data_dir + "/HGG/") ]
filenames.extend([ data_dir + "/LGG/" + f + "/"\
   for f in os.listdir(data_dir + "/LGG/") ])

filenames = [ f + d for f in filenames for d in os.listdir(f) ]

t1_l = sorted([ f for f in filenames if "t1.nii.gz" in f ])
t1ce_l = sorted([ f for f in filenames if "t1ce.nii.gz" in f ])
t2_l = sorted([ f for f in filenames if "t2.nii.gz" in f ])
flair_l = sorted([ f for f in filenames if "flair.nii.gz" in f ])
segs_l = sorted([ f for f in filenames if "seg.nii.gz" in f ])

def replace_dir_name(orig_dir):
  return orig_dir.replace('brats2018', 'brats2018downsampled')

for t1, t1ce, t2, flair, seg in zip(t1_l, t1ce_l, t2_l, flair_l, segs_l):
  t1_img = nib.load(t1)
  t1ce_img = nib.load(t1ce)
  t2_img = nib.load(t2)
  flair_img = nib.load(flair)
  seg_img = nib.load(seg)

  aff = t1_img.affine.copy()
  outshape = tuple([int(float(x)/scaling) for x in t1_img.shape])
  print(t1ce_img.shape)
  print(outshape)
  aff[:3, :3]*=scaling
  t1_out =\
      resample_img(t1_img, target_affine=aff, target_shape=outshape, interpolation='nearest')
  t1ce_out =\
      resample_img(t1ce_img, target_affine=aff, target_shape=outshape, interpolation='nearest')
  t2_out =\
      resample_img(t2_img, target_affine=aff, target_shape=outshape, interpolation='nearest')
  flair_out =\
      resample_img(flair_img, target_affine=aff, target_shape=outshape, interpolation='nearest')
  seg_out =\
      resample_img(seg_img, target_affine=aff, target_shape=outshape, interpolation='nearest')

  for f, d in [(t1, t1_out), (t1ce, t1ce_out), (t2, t2_out), (flair, flair_out), (seg, seg_out)]:
    out_file = replace_dir_name(f)
    split_file = out_file.split('/')
    out_dirs = "/".join(split_file[:-1])
    os.makedirs(out_dirs, exist_ok=True)
    d.to_filename(out_file)


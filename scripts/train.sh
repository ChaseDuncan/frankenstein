#!/bin/bash

# Copy unzip dataset from /shared/rsaas/cddunca2 to /data/cddunca2,
# the node's local storage.

cp /shared/rsaas/cddunca2/MICCAI_BraTS_2018_Data_Training.zip /data/cddunca2/
mkdir /data/cddunca2/brats/
unzip data/cddunca2/MICCAI_BraTS_2018_Data_Training.zip -d /data/cddunca2/brats/

python train.py --config ./config/basic.cfg --checkpoint_dir ./bilinear/ --upsampling bilinear

rm -rf /data/cddunca2/brats/

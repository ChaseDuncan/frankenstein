# Make directory on server and unzip data.

rm -rf /data/cddunca2/brats2018/

mkdir -p /data/cddunca2/brats2018
mkdir -p /data/cddunca2/brats2018/training
mkdir -p /data/cddunca2/brats2018/validation

cp /shared/rsaas/cddunca2/MICCAI_BraTS_2018_Data_Training.zip /data/cddunca2/
cp /shared/rsaas/cddunca2/MICCAI_BraTS_2018_Data_Validation.zip /data/cddunca2/

unzip /data/cddunca2/MICCAI_BraTS_2018_Data_Training.zip -d /data/cddunca2/brats2018/training
unzip /data/cddunca2/MICCAI_BraTS_2018_Data_Validation.zip -d /data/cddunca2/brats2018/validation


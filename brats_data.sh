# Make directory on server and unzip data.

mkdir -p /data/cddunca2/brats2018
cp /shared/rsaas/cddunca2/MICCAI_BraTS_2018_Data_Training.zip /data/cddunca2/
unzip /data/cddunca2/MICCAI_BraTS_2018_Data_Training.zip -d /data/cddunca2/brats2018/

# Multimodal MRI Segmentation Experiments
An experimental framework for evaluating the performance for various
techniques for segmenting the BraTS 2018 data.

## How to run:

These instructions are for how to build and run this framework on an
Azure Ubuntu VM which is the environment in which this work was originally
done.

The BraTS 2018 data is required to run this module. The data can be
found [here](https://ipp.cbica.upenn.edu/). It is required to make an
account and request the data.

Download and install [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

Install Nvidia toolkit:
```
sudo apt-get install nvidia-cuda-toolkit 
```
Install PyTorch:

```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

Note that this will install torch for Cuda 9.0. You should check that this
is the correct version by running 

``` 
nvcc --version 
```



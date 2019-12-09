![](https://github.com/neuropoly/ivado-medical-imaging/workflows/Python%20package/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/neuropoly/ivado-medical-imaging/blob/cg/organize-repo/LICENSE.md)
# IVADO Medical Imaging
Comprehensive and open-source repository of deep learning methods for medical data segmentation.
Collaboration between MILA and NeuroPoly for the IVADO project on medical imaging.

- [Installing](#installing)
- [Contributions](#contributions-and-features)
- [Training](#training)
- [Data](#data)

## Contributions and features

### Physic-inform network
We adapted the Feature-wise Linear Modulation ([FiLM](https://arxiv.org/pdf/1709.07871.pdf)) approach to the segmentation task. FiLM enabled us to modulate CNNs features based on non-image metadata.

![Figure FiLM](/images/film_figure.png)

### Two-step training with class sampling
We implemented a two-step training scheme, using class sampling, in order to mitigate the issue of class imbalance. During the first step, the CNN is trained on an equivalent proportion of positive and negative samples, negative samples being under-weighted during the data loading at each epoch. During the second step, the CNN is fine-tuned on the realistic (i.e. class-imbalanced) dataset.

## mixup
XX [mixup](https://arxiv.org/pdf/1710.09412.pdf)

## Data augmentation on lesion ground-truths
XX

### Network architectures
- [U-net](https://arxiv.org/pdf/1505.04597.pdf)
- [HeMIS](https://arxiv.org/abs/1607.05194)

### Loss functions
- [Dice Loss](https://arxiv.org/abs/1606.04797)
- [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf)
- [Generalised Dice Loss](https://arxiv.org/pdf/1707.03237.pdf)

## Installing

This project requires Python 3.6 and PyTorch >= 1.2.0. We recommend you work under a virtual environment:

~~~
virtualenv venv-ivadomed --python=python3.6
source venv-ivadomed/bin/activate
~~~

Then, install all requirements using `pip`:

```
git clone https://github.com/neuropoly/ivado-medical-imaging.git
cd ivado-medical-imaging
pip install -e .
```

## Training

To train the network, use the `ivadomed` command-line tool that will be available on your path after installation, example below:

```
ivadomed config/config.json
```

where `config.json` is a configuration file. A description of each parameter is available in the [wiki](https://github.com/neuropoly/ivado-medical-imaging/wiki/configuration-file).


## Data

The working dataset are:
1. derived from the [Spinal Cord MRI Public Database](https://openneuro.org/datasets/ds001919).
2. the spinal cord grey matter segmentation [challenge dataset](https://www.sciencedirect.com/science/article/pii/S1053811917302185#s0050).
3. private multi-center dataset (`duke/sct_testing/large`) for spinal cord and MS lesion segmentation task.

The data structure is compatible with [BIDS](http://bids.neuroimaging.io/) and is exemplified below:
~~~
bids_folder/
└── dataset_description.json
└── participants.tsv
└── sub-amu01
    └── anat
        └── sub-amu01_T1w_reg.nii.gz --> Processed (i.e. different than in the original SpineGeneric database)
        └── sub-amu01_T1w_reg.json
        └── sub-amu01_T2w_reg.nii.gz --> Processed
        └── sub-amu01_T2w_reg.json
        └── sub-amu01_acq-MTon_MTS_reg.nii.gz --> Processed
        └── sub-amu01_acq-MTon_MTS_reg.json
        └── sub-amu01_acq-MToff_MTS_reg.nii.gz --> Processed
        └── sub-amu01_acq-MToff_MTS_reg.json
        └── sub-amu01_acq-T1w_MTS.nii.gz --> Unprocessed (i.e. same as in the original SpineGeneric database)
        └── sub-amu01_acq-T1w_MTS.json
        └── sub-amu01_T2star_reg.nii.gz --> Processed
        └── sub-amu01_T2star_reg.json
└── derivatives
    └── labels
        └── sub-amu01
            └── anat
                └── sub-amu01_T1w_seg.nii.gz --> Spinal cord segmentation
~~~

## Contributors
[List of contributors](https://github.com/neuropoly/ivado-medical-imaging/graphs/contributors)

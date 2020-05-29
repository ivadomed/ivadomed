[![Coverage Status](https://coveralls.io/repos/github/neuropoly/ivado-medical-imaging/badge.svg?branch=master)](https://coveralls.io/github/neuropoly/ivado-medical-imaging?branch=master)
![](https://github.com/neuropoly/ivado-medical-imaging/workflows/Python%20package/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

<p float="left">
  <img src="https://github.com/neuropoly/ivado-medical-imaging/raw/master/images/neuropoly_logo.png" height="120" />
  <img src="https://github.com/neuropoly/ivado-medical-imaging/raw/master/images/mila_logo.png" height="120" />
  <img src="https://github.com/neuropoly/ivado-medical-imaging/raw/master/images/ivado_logo.png" height="120" />
</p>

# IVADO Medical Imaging
Comprehensive and open-source repository of deep learning methods for medical data segmentation.
Collaboration between MILA and NeuroPoly for the IVADO project on medical imaging.

- [Installing](#installing)
- [Contributions](#contributions-and-features)
- [Training](#training)
- [Data](#data)

## Installing


This project requires Python 3.6 and PyTorch >= 1.5.0. We recommend you work under a virtual environment:

~~~
virtualenv venv-ivadomed --python=python3.6
source venv-ivadomed/bin/activate
~~~

### Option 1 : development version from Github
ivadomed is installed from Github and the requirements are installed using `pip`:

```
git clone https://github.com/neuropoly/ivado-medical-imaging.git
cd ivado-medical-imaging
pip install -e .
```

### Option 2 : release from PyPI

ivadomed and its requirements are installed directly using `pip` :

```
pip install --upgrade pip
pip install ivadomed
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

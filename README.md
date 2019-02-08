# IVADO Medical Imaging
This is a repository for the collaboration between MILA and NeuroPoly for the IVADO project on medical imaging.

## Installing
This project requires Python 3.6 and PyTorch >= 1.0, to install all requirements, please use `pip` as described below:

```
~$ git clone https://github.com/neuropoly/ivado-medical-imaging.git
~$ cd ivado-medical-imaging
~$ pip install -e .
```

And all dependencies will be installed into your own system.

## Data
The working dataset is derived from the [Spinal Cord MRI Public Database](https://osf.io/76jkx/)

The data structure is compatible with [BIDS](http://bids.neuroimaging.io/) and is exemplified below:
~~~
site/
└── dataset_description.json
└── participants.tsv
└── sub-01
    └── anat
             └── sub-01_T1w_reg.nii.gz --> Processed (i.e. different than in the original SpineGeneric database)
             └── sub-01_T1w_reg.json
             └── sub-01_T2w_reg.nii.gz --> Processed
             └── sub-01_T2w_reg.json
             └── sub-01_acq-MTon_MTS_reg.nii.gz --> Processed
             └── sub-01_acq-MTon_MTS_reg.json
             └── sub-01_acq-MToff_MTS_reg.nii.gz --> Processed
             └── sub-01_acq-MToff_MTS_reg.json
             └── sub-01_acq-T1w_MTS.nii.gz --> Unprocessed (i.e. same as in the original SpineGeneric database)
             └── sub-01_acq-T1w_MTS.json
             └── sub-01_T2star_reg.nii.gz --> Processed
             └── sub-01_T2star_reg.json
    └── dwi
             └── sub-01_dwi.nii.gz
             └── sub-01_dwi.bval
             └── sub-01_dwi.bvec
             └── sub-01_dwi.json
└── derivatives
    └── labels
        └── sub-01
            └── anat
                └── sub-01_T1w_seg.nii.gz --> Spinal cord segmentation
~~~


## Training
To train the network, use the `ivadomed` command-line tool that will be available on your path after installation, example below:

```
ivadomed config.json
```

The `config.json` is a configuration example.

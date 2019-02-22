# IVADO Medical Imaging
This is a repository for the collaboration between MILA and NeuroPoly for the IVADO project on medical imaging.

## Installing
This project requires Python 3.6 and PyTorch >= 1.0.1, to install all requirements, please use `pip` as described below:

```
~$ git clone https://github.com/neuropoly/ivado-medical-imaging.git
~$ cd ivado-medical-imaging
~$ pip install -e .
```

And all dependencies will be installed into your own system.

## Training
To train the network, use the `ivadomed` command-line tool that will be available on your path after installation, example below:

```
ivadomed configs/config.json
```

The `config.json` is a configuration example. During the training, you can open TensorBoard and it will show the following statistics and visualization:

### TensorBoard - Validation Metrics
These are the metrics computed for the validation dataset. It contains results for pixel-wise accuracy, Dice score, mIoU (mean intersection over union), pixel-wise precision, recall and specificity.
![](/images/validation_metrics.png)

### TensorBoard - Training set samples
These are visualizations of the training samples (after data augmentation), their ground truths and predictions from the network.
![](/images/train_vis.png)

### TensorBoard - Validation set samples
These are visualizations of the validation samples, their ground truths and predictions from the network.
![](/images/validation_vis.png)

### TensorBoard - Losses
This is the visualization of the losses during the training (using 50 epochs in that example).
![](/images/losses.png)

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
└── derivatives
    └── labels
        └── sub-01
            └── anat
                └── sub-01_T1w_seg.nii.gz --> Spinal cord segmentation
~~~



## Contributors
[List of contributors](https://github.com/neuropoly/ivado-medical-imaging/graphs/contributors)

## License

The MIT License (MIT)

Copyright (c) 2019 Polytechnique Montreal, Université de Montréal

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# IVADO Medical Imaging
This is a repository for the collaboration between MILA and NeuroPoly for the IVADO project on medical imaging.

- [Installing](#installing)
- [Training](#training)
- [Baseline Results](#baseline-results)
- [Data](#data)


## Installing

This project requires Python 3.6 and PyTorch >= 1.0.1. We recommend you work under a virtual environment:

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

## Baseline results
If you use the `configs/config.json` file for training, it should produce the following metrics in the evaluation set:

| Model    | Accuracy | Dice  | Haussdorf | mIoU  | Precision | Recall | Specificity |
|----------|----------|-------|-----------|-------|-----------|--------|-------------|
| Baseline | 99.85    | 95.42 | 1.193     | 91.67 | 94.92     | 96.07  | 99.92       |

For more details on the meaning of the evaluation metrics, please see [Prados et al. (look for: Validation Metrics)](https://www.sciencedirect.com/science/article/pii/S1053811917302185#s0050).

## Data

The working dataset is derived from the [Spinal Cord MRI Public Database](https://osf.io/76jkx/). The processed dataset that should be used for training the model is available [here](https://osf.io/agp3q/). When training a model, make sure to indicate the [checksum SH2](https://osf.io/agp3q/?show=revision) inside the `config.json` file.

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

### Data processing

The generation of labels is done automatically using the [Spinal Cord Toolbox](https://github.com/neuropoly/spinalcordtoolbox). The processing scripts are located in `prepare_data/`.

## Contributors
[List of contributors](https://github.com/neuropoly/ivado-medical-imaging/graphs/contributors)

## License

The MIT License (MIT)

Copyright (c) 2019 Polytechnique Montreal, Université de Montréal

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

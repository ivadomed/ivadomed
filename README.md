# IVADO Medical Imaging
This is a repository for the collaboration between MILA and NeuroPoly for the IVADO project on medical imaging.

- [Installing](#installing)
- [Training](#training)
- [Baseline Results](#baseline-results)
- [Data](#data)


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

The `config.json` is a configuration example.
Please find below a description of each parameter:

#### Basic parameters and hyperparameters

- `command`: run the specified command (choice: ``"train"``, ``"test"``).
- `gpu`: ID of the GPU to use.
- `target_suffix`: suffix of the derivative file containing the ground-truth of interest (e.g. `"_seg-manual"`, `"_lesion-manual"`).
- `roi_suffix`: suffix of the derivative file containing the ROI used to crop (e.g. `"_seg-manual"`) with `ROICrop2D` as transform. Please use `null` if you do not want to use a ROI to crop (ie use `CenterCrop2D`).
- `bids_path`: relative path of the BIDS folder.
- `random_seed`: seed used by the random number generator to split the dataset between training/validation/testing.
- `contrast_train_validation`: list of image modalities included in the training and validation datasets.
- `contrast_balance`: used to under-represent some modalities in the dataset (e.g. `{"T1w": 0.1}` will include only 10% of the available `T1w` images into the training/validation/test set).
- `contrast_test`: list of image modalities included in the testing dataset.
- `batch_size`: int.
- `dropout_rate`: float (e.g. 0.4).
- `batch_norm_momentum`: float (e.g. 0.1).
- `num_epochs`: int.
- `initial_lr`: initial learning rate.
- `schedule_lr`: method of learning rate scheduling. Choice between: `"CosineAnnealingLR"`, `"CosineAnnealingWarmRestarts"` and `"CyclicLR"`. Please find documentation [here](https://pytorch.org/docs/stable/optim.html).
- `loss`: dictionary with a key `"name"` for the choice between `"dice"`, `"focal"`, `"focal_dice"`, `"gdl"` and `"cross_entropy"` and a (optional) key `"params"` (e.g.`{"name": "focal", "params": {"gamma": 0.5}}`.
- `log_directory`: folder name where log files are saved.
- `debugging`: allows extended verbosity and intermediate outputs (choice: `false` or `true`).
- `split_path`: (optional) path to joblib file containing the list of training / validation / test subjects, used to ensure reproducible experiments
- `early_stopping_epsilon`: Threshold (percentage) for an improvement in the validation loss the be considered meaningful
- `early_stopping_patience`: Number of epochs after which the training is stopped if the validation loss improvement not meaningful (less than `early_stopping_epsilon`)

#### Network architecture
- `film_layers`: indicates on which layer(s) of the U-net you want to apply a FiLM modulation: list of 8 elements (because Unet has 8 layers), set to `0` for no FiLM modulation, set `1` otherwise. Note: When running `Unet` or `MixedUp-Unet`, please fill this list with zeros only.
- `metadata`: choice between `"without"`, `"mri_params"`, and `"contrast"`.
If `"mri_params"`, then vectors of [FlipAngle, EchoTime, RepetitionTime, Manufacturer] are input to the FiLM generator. If `"contrast"`, then image contrasts (according to `config/contrast_dct.json`) are input to the FiLM generator. Notes:
    - If `"mri_params"`, then only images with TR, TE, FlipAngle, and Manufaturer available info are included.
    - Please use '"without"' when comparing `Unet` vs. `MixedUp-Unet` ; otherwise compare `Unet` vs. `FiLMed-Unet`.
- `mixup_bool`: indicates if mixup is applied to the training data (choice: `false` or `true`). Note: Please use `false` when comparing `Unet` vs. `FiLMed-Unet`.
- `mixup_alpha`: alpha parameter of the Beta distribution (float).


#### Loader preprocessing
- `slice_axis`: choice between `"sagittal"`, `"coronal"`, and `"axial"`. This parameter decides the slice orientation on which the model will train.
- `balance_samples`: choice between `true` and `false`. If `true`, then positive and negative GT samples are balanced in both training and validation datasets.
- `slice_filter`:
    1. `filter_empty_input`: choice between `true` or `false`, filter empty images if `true`
    2. `filter_empty_mask`: choice between `true` and `false`, filter empty GT mask slices if `true`
- `slice_filter_roi`: int, it filters ROI mask slices with less than this threshold of non zero voxels. Active when using `"ROICrop2D"` and inputing ROI file.
- `split_method`: choice between `"per_patient"` or `"per_center"`.
- `train_fraction`: number between `0` and `1` representing the fraction of the dataset used as training set.
- `test_fraction`: number between `0` and `1` representing the fraction of the dataset used as test set. This parameter is only used if the `split_method` is `"per_patient"`.
- `data_augmentation_training`: this parameter is a dictionnary containing the training transformations. The transformation are
    - `ToTensor` (parameters: `labeled`)
    - `CenterCrop2D` (parameters: `size`, `labeled`)
    - `ROICrop2D` (parameters: `size`, `labeled`)
    - Normalize` (parameters: `mean`, `std`)
    - `NormalizeInstance` (parameters: `sample`)
    - `RandomRotation` (parameters: `degrees`, `resample`, `expand`, `center`, `labeled`)
    - `RandomAffine` (parameters: `degrees`, `translate`, `scale`, `shear`, `resample`, `fillcolor`, `labeled`)
    - `RandomTensorChannelShift` (parameters: `shift_range`)
    - `ElasticTransform` (parameters: `alpha_range`, `sigma_range`, `p`, `labeled`)
    - `Resample` (parameters: `wspace`, `hspace`, `interpolation`, `labeled`)
    - `AdditionGaussianNoise` (parameters: `mean`, `std`)
    - `DilateGT` (parameters: `dilation_factor`) float, controls the number of iterations of ground-truth dilation depending on the size of each individual lesion, data augmentation of the training set. Use `0` to disable.
    - For more information on these transformations and their parameters see documentation [here](https://github.com/perone/medicaltorch/blob/master/medicaltorch/transforms.py).
- `data_augmentation_validation`: this parameter is a dictionnary containing the validation/test transformations. The choices are the same as `data_augmentation_training`.



## References

Please find below the original articles of methods we implemented in this project:
- [U-net](https://arxiv.org/pdf/1505.04597.pdf)
- [FiLM](https://arxiv.org/pdf/1709.07871.pdf)
- [mixup](https://arxiv.org/pdf/1710.09412.pdf)
- [focal loss](https://arxiv.org/pdf/1708.02002.pdf)
- [GT dilation -- data augmentation](https://arxiv.org/pdf/1901.09263.pdf)
- [Generalised Dice Loss](https://arxiv.org/pdf/1707.03237.pdf)

During the training, you can open TensorBoard and it will show the following statistics and visualization:

```
tensorboard --logdir=./log_sc
```
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

The working dataset are:
1. derived from the [Spinal Cord MRI Public Database](https://openneuro.org/datasets/ds001919).
2. the spinal cord grey matter segmentation [challenge dataset](https://www.sciencedirect.com/science/article/pii/S1053811917302185#s0050).
3. private multi-center dataset (`duke/sct_testing/large`).

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

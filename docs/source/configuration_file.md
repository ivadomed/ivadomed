# Configuration file

Examples of configuration files: [here](https://github.com/neuropoly/ivado-medical-imaging/tree/master/ivadomed/config)

* [General parameters](#general-parameters)
* [Loader parameters](#loader-parameters)
* [Split dataset](#split-dataset)
* [Training parameters](#training-parameters)
* [Architecture](#architecture)
* [Testing parameters](#testing-parameters)
* [Transformations](#transformations)

## General parameters
#### command
Run the specified command. Choices: ``"train"``, ``"test"``, ``"eval"``, to train, test and evaluate a model respectively.
#### gpu
Integer. ID of the GPU to use.
#### log_directory
Folder name where output files (e.g. trained model, predictions, results) are saved.
#### debugging
Bool. Allows extended verbosity and intermediate outputs.

## Loader parameters
#### bids_path
String. Relative path of the BIDS folder.
#### target_suffix
Suffix list of the derivative file containing the ground-truth of interest (e.g. [`"_seg-manual"`, `"_lesion-manual"`]). If the list has a length greater than 1, then a multi-class model will be trained.
#### contrasts
- `train_validation`: list of image modalities loaded for the training and validation. If `multichannel`, then this list represent the different channels of the input tensors. Otherwise, the modalities are mixed and the model has only one input channel.
- `test`: list of image modalities loaded in the testing dataset. Same comment than for `train_validation` regarding `multichannel`.
- `balance`: Dict. Used to under-represent some modalities in the dataset: e.g. `{"T1w": 0.1}` will include only 10% of the available `T1w` images into the training/validation/test set. Please set `multichannel` to `false` if you are using this parameter.
#### multichannel
Bool. Indicated if more than a modality is used by the model. See details in both `train_validation` and `test` for the modalities that are input.
#### slice_axis
Choice between `"sagittal"`, `"coronal"`, and `"axial"`.
This parameter decides the slice orientation on which the model will train.
#### slice_filter
Dict:
- `filter_empty_input`: Bool. Filter empty images if `true`.
- `filter_empty_mask`: Bool. Filter empty GT mask slices if `true`.
#### roi
Dict. of parameters about the region of interest
- `suffix`: suffix of the derivative file containing the ROI used to crop (e.g. `"_seg-manual"`) with `ROICrop` as transform. Please use `null` if you do not want to use a ROI to crop.
- `slice_filter_roi`: int. It filters ROI mask slices with less than this threshold of non zero voxels. Active when using `"ROICrop"` and inputing ROI file.

## Split dataset
#### fname_split
Filename of a joblib file containing the list of training / validation / test subjects, used to ensure reproducible experiments
#### random_seed
Seed used by the random number generator to split the dataset between training/validation/testing.
#### center_test
List. List of centers to only include in the testing dataset.
#### method
Choice between `"per_patient"` (i.e. shuffle all subjects then splits) or `"per_center"` (split subject according to their acquisition centers).
#### train_fraction
Number between `0` and `1` representing the fraction of the dataset used as training set.
#### test_fraction
Number between `0` and `1` representing the fraction of the dataset used as test set. This parameter is only used if the `method` is `"per_patient"`.

## Training parameters
#### batch_size
Integer.
#### loss
- `name`: Name of the loss function: Choice among the classes that are available [here](https://github.com/neuropoly/ivado-medical-imaging/blob/master/ivadomed/losses.py).
- Other parameters that could be needed in the Loss function definition: see attributes of the Loss function of interest (e.g. `"gamma": 0.5` for `FocalLoss`).
#### training_time
- `num_epochs`: int.
- `early_stopping_epsilon`: Threshold (percentage) for an improvement in the validation loss to be considered meaningful
- `early_stopping_patience`: Number of epochs after which the training is stopped if the validation loss improvement not meaningful (i.e. less than `early_stopping_epsilon`)
#### scheduler
- `initial_lr`: Float. Initial learning rate.
- `scheduler_lr`:
1. `name`: Choice between: `"CosineAnnealingLR"`, `"CosineAnnealingWarmRestarts"` and `"CyclicLR"`. Please find documentation [here](https://pytorch.org/docs/stable/optim.html).
2. Other parameters that are needed for the scheduler of interest (e.g. `"base_lr": 1e-5, "max_lr": 1e-2` for `"CosineAnnealingLR"`).
#### balance_samples
Bool. If `true`, then positive and negative GT samples are balanced in both training and validation datasets.
#### mixup_alpha
Float. Alpha parameter of the Beta distribution, see [original paper](https://arxiv.org/pdf/1710.09412.pdf).
#### transfer_learning
- `retrain_model`: Filename of the pretrained model (`path/to/pretrained-model`). If `null`, then no transfer learning is performed and the network is trained from scratch.
- `retrain_fraction`: Float between 0. and 1. Controls the fraction of the pre-trained model that will be fine-tuned. For instance, if set to 0.5, then the second half of the model will be fine-tuned, while the first layers will be frozen.

## Architecture
Architectures for both segmentation and classification are available and described [here](models.rst).
If the selected architecture is listed [here](https://github.com/neuropoly/ivado-medical-imaging/blob/master/ivadomed/loader/loader.py#L14), then a classification task is run, ie the ground-truth are labels extracted from `target`, instead of arrays for the segmentation task.
### default_model (Mandatory)
Define the default model (`Unet`) and mandatory parameters that are common to all available architectures (listed [here](models.rst)). If a tailored model is defined (see next section), then the default parameters are merged with the parameters that are specific to the tailored model.
- `name`: `Unet` (default)
- `dropout_rate`: float (e.g. 0.4).
- `batch_norm_momentum`: float (e.g. 0.1).
- `depth`: int, number of down-sampling operations.
Note:
- `in_channel` is automatically defined with `multichannel` and `contrast/training_validation`.
- `out_channel` is automatically defined with `target_suffix`.
### Tailored model (optional)
Here are defined the tailored model and the parameters that are specific to it (ie not defined in the default model). See examples:
- [FiLMedUnet](https://github.com/neuropoly/ivado-medical-imaging/blob/master/ivadomed/config/config.json#L64)
    - `metadata`: choice between `"without"`, `"mri_params"`, and `"contrast"`. If `"mri_params"`, then vectors of [FlipAngle, EchoTime, RepetitionTime, Manufacturer] are input to the FiLM generator. If `"contrast"`, then image contrasts (according to `config/contrast_dct.json`) are input to the FiLM generator.
- [HeMISUnet](https://github.com/neuropoly/ivado-medical-imaging/blob/master/ivadomed/config/config_spineGeHemis.json#L64)
    - `missing_modality`: Bool.
- [UNet3D](https://github.com/neuropoly/ivado-medical-imaging/blob/master/ivadomed/config/config_tumorSeg.json#L65)
    - `length_3D`: tuple indicating the size of the subvolumes or volume used for unet 3D model (depth, width, height).
    - `padding_3D`: size of the overlapping per subvolume and dimensions (e.i `padding:0`). Note: In order to be used, each dimension of an input image needs to be a multiple of length plus 2 * padding and a multiple of 16. To change input image size use the following transformation `CenterCrop3D`. 
    - `attention_unet`: indicates if attention gates are used in the Unet's decoder.
## Testing parameters
- `binarize_prediction`: Indicates if predictions from the model are soft (prediction from 0 to 1) or binarized (either 1 or 0 according to the threshold 0.5).
#### uncertainty
Uncertainty computation is performed if `n_it>0` and at least `epistemic` or `aleatoric` is `True`.
- `epistemic`: Bool.
- `aleatoric`: Bool.
- `n_it`: int, number of Monte Carlo iterations.

## Transformations
Indicate the transformation in the same order you would like them to be applied to your samples. For each transformation, please indicate the parameters as well as the following two optional entries:
- `applied_to`: list betweem `"im", "gt", "roi"`. If not specified, then the transformation is applied to all loaded samples. Otherwise, only applied to the specified types: eg `["gt"]` implies that this transformation is only applied to the ground-truth data.
- `dataset_type`: list between `"training", "validation", "testing"`. If not specified, then the transformation is applied to the three sub-datasets. Otherwise, only applied to the specified subdatasets: eg `["testing"]` implies that this transformation is only applied to the testing sub-dataset.
### Available transformations:
- `NumpyToTensor`
- `CenterCrop2D` (parameters: `size`)
- `ROICrop2D` (parameters: `size`)
- `NormalizeInstance`
- `RandomRotation` (parameters: `degrees`)
- `RandomAffine` (parameters: `translate`)
- `RandomShiftIntensity` (parameters: `shift_range`)
- `ElasticTransform` (parameters: `alpha_range`, `sigma_range`, `p`)
- `Resample` (parameters: `wspace`, `hspace`, `dspace`)
- `AdditionGaussianNoise` (parameters: `mean`, `std`)
- `DilateGT` (parameters: `dilation_factor`) float, controls the number of iterations of ground-truth dilation depending on the size of each individual lesion, data augmentation of the training set. Use `0` to disable.
- `HistogramClipping` (parameters: `min_percentile`, `max_percentile`)
- `Clage` (parameters: `clip_limit`, `kernel_size`)
- ` RandomReverse`
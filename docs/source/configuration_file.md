# Configuration file

## General parameters

### command
Run the specified command. Choices: ``"train"``, ``"test"``, ``"eval"``, to train, test and evaluate a model respectively.

#### gpu
Integer. ID of the GPU to use.

#### log_directory
Folder name that will contain the output files (e.g., trained model, predictions, results).

#### debugging
Bool. Extended verbosity and intermediate outputs.

## Loader parameters

#### bids_path
String. Path of the BIDS folder.

#### target_suffix
Suffix list of the derivative file containing the ground-truth of interest (e.g. [`"_seg-manual"`, `"_lesion-manual"`]). If the list has a length greater than 1, then a multi-class model will be trained.

#### contrasts
- `train_validation`: list of image contrasts (e.g. `T1w`, `T2w`) loaded for the training and validation. If `multichannel`, then this list represent the different channels of the input tensors. Otherwise, the contrasts are mixed and the model has only one input channel.
- `test`: list of image contrasts (e.g. `T1w`, `T2w`) loaded in the testing dataset. Same comment than for `train_validation` regarding `multichannel`.
- `balance`: Dict. Enables to weight the importance of specific channels (or contrasts) in the dataset: e.g. `{"T1w": 0.1}` means that only 10% of the available `T1w` images will be included into the training/validation/test set. Please set `multichannel` to `false` if you are using this parameter.

#### multichannel
Bool. Indicated if more than a contrast (e.g. `T1w` and `T2w`) is used by the model. See details in both `train_validation` and `test` for the contrasts that are input.

#### slice_axis
Choice between `"sagittal"`, `"coronal"`, and `"axial"`.
Sets the slice orientation for on which the model will be used.

#### slice_filter
Dict. Discard a slice from the dataset if it meets a condition, see below.
- `filter_empty_input`: Bool. Discard slices where all voxel intensities are zeros.
- `filter_empty_mask`: Bool. Discard slices where all voxel labels are zeros.

#### roi
Dict. of parameters about the region of interest
- `suffix`: String. Suffix of the derivative file containing the ROI used to crop (e.g. `"_seg-manual"`) with `ROICrop` as transform. Please use `null` if you do not want to use a ROI to crop.
- `slice_filter_roi`: int. It filters (i.e. discards from the dataset) slices where the ROI mask has with less than this number of non zero voxels. Active when using `"ROICrop"`.

## Split dataset

#### fname_split
Filename of a joblib file containing the list of training/validation/testing subjects. This file can later be used to re-train a model using the same data splitting scheme.

#### random_seed
Seed used by the random number generator to split the dataset between training/validation/testing.

#### center_test
List of strings. List of centers to only include in the testing dataset. If used, please include a column `institution_id` in your `bids_dataset/participants.tsv`.

#### method
Choice between `"per_patient"` (i.e. shuffle all subjects then splits, using the `participant_id` column from `my_bids_dataset/participants.tsv`) or `"per_center"` (split subjects according to their acquisition centers, using the `institution_id` column from `my_bids_dataset/participants.tsv`).

#### train_fraction
Float. Between `0` and `1` representing the fraction of the dataset used as training set.

#### test_fraction
Float. Between `0` and `1` representing the fraction of the dataset used as test set. This parameter is only used if the `method` is `"per_patient"`.

## Training parameters

#### batch_size
Strictly positive integer.

#### loss
- `name`: Name of the loss function: Choice among the classes that are available [here](https://ivadomed.org/en/latest/api_ref.html#ivadomed-losses).
- Other parameters that could be needed in the Loss function definition: see attributes of the Loss function of interest (e.g. `"gamma": 0.5` for `FocalLoss`).

#### training_time
- `num_epochs`: Strictly positive integer.
- `early_stopping_epsilon`: Float. If the validation loss difference during one epoch (i.e. `abs(validation_loss[n] - validation_loss[n-1]` where n is the current epoch) is inferior to this epsilon for `early_stopping_patience` consecutive epochs, then  the early stopping of the training is triggered.
- `early_stopping_patience`: Strictly positive integer. Number of epochs after which the training is stopped if the validation loss improvement is smaller than `early_stopping_epsilon`.

#### scheduler
- `initial_lr`: Float. Initial learning rate.
- `scheduler_lr`:
1. `name`: Choice between: `"CosineAnnealingLR"`, `"CosineAnnealingWarmRestarts"` and `"CyclicLR"`. Please find documentation [here](https://pytorch.org/docs/stable/optim.html).
2. Other parameters that are needed for the scheduler of interest (e.g. `"base_lr": 1e-5, "max_lr": 1e-2` for `"CosineAnnealingLR"`).

#### balance_samples
Bool. Balance positive and negative labels in both the training and the validation datasets.

#### mixup_alpha
Float. Alpha parameter of the Beta distribution, see [original paper on the Mixup technique](https://arxiv.org/abs/1710.09412).

#### transfer_learning
- `retrain_model`: Filename of the pretrained model (`path/to/pretrained-model`). If `null`, no transfer learning is performed and the network is trained from scratch.
- `retrain_fraction`: Float between 0. and 1. Controls the fraction of the pre-trained model that will be fine-tuned. For instance, if set to 0.5, the second half of the model will be fine-tuned while the first layers will be frozen.

## Architecture
Architectures for both segmentation and classification are available and described in the [Models](models.rst) section.
If the selected architecture is listed in the [loader.py](../../ivadomed/loader/loader.py#L14) file, a classification (not segmentation) task is run. In the case of a classification task, the ground truth will correspond to a single label value extracted from `target`, instead being an array (the latter being used for the segmentation task).

### default_model (Mandatory)
Define the default model (`Unet`) and mandatory parameters that are common to all available architectures (listed in the [Models](models.rst) section). If a tailored model is defined (see next section), the default parameters are merged with the parameters that are specific to the tailored model.
- `name`: `Unet` (default)
- `dropout_rate`: float (e.g. 0.4).
- `batch_norm_momentum`: Float (e.g. 0.1).
- `depth`: Strictly positive integer. Number of down-sampling operations.
Note:
- `in_channel` is automatically defined with `multichannel` and `contrast/training_validation`.
- `out_channel` is automatically defined with `target_suffix`.

### Tailored model (optional)
Here are defined the tailored model and the parameters that are specific to it (ie not defined in the default model). See examples:
- `FiLMedUnet`
    - `metadata`: `{'without', 'mri_params', 'contrast'}`. `mri_params`: Vectors of [FlipAngle, EchoTime, RepetitionTime, Manufacturer] are input to the FiLM generator. `contrast`: Image contrasts (according to `config/contrast_dct.json`) are input to the FiLM generator.
- `HeMISUnet`
    - `missing_contrast`: Bool.
- `UNet3D`
    - `length_3D`: (Integer, Integer, Integer). Size of the subvolumes or volume used for unet 3D model: (depth, width, height).
    - `padding_3D`: size of the overlapping per subvolume and dimensions (e.i `padding:0`). Note: In order to be used, each dimension of an input image needs to be a multiple of length plus 2 * padding and a multiple of 16. To change input image size use the following transformation `CenterCrop3D`. 
    - `attention_unet`: Bool. Use attention gates in the Unet's decoder.

## Testing parameters
- `binarize_prediction`: Bool. Binarize output predictions using a threshold of 0.5. If 'false', output predictions are float between 0 and 1. 

#### uncertainty
Uncertainty computation is performed if `n_it>0` and at least `epistemic` or `aleatoric` is `True`. Note: both `epistemic` and `aleatoric` can be `true`.
- `epistemic`: Bool. Model-based uncertainty with [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142).
- `aleatoric`: Bool. Image-based uncertainty with [test-time augmentation](https://doi.org/10.1016/j.neucom.2019.01.103).
- `n_it`: Integer. Number of Monte Carlo iterations. Set to 0 for no uncertainty computation.

## Transformations
Transformations applied during data augmentation. Transformations are sorted in the order they are applied to the image samples. For each transformation, the following parameters are customizable:
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
- `DilateGT` (parameters: `dilation_factor`) Float. Controls the number of iterations of ground-truth dilation depending on the size of each individual lesion, data augmentation of the training set. Use `0` to disable.
- `HistogramClipping` (parameters: `min_percentile`, `max_percentile`)
- `Clage` (parameters: `clip_limit`, `kernel_size`)
- `RandomReverse`

## Examples
Examples of configuration files: [here](../../ivadomed/config).

In particular:
- [config_classification.json](../../ivadomed/config/config_classification.json) is dedicated to classification task.
- [config_sctTesting.json](../../ivadomed/config/config_sctTesting.json) is a user case of 2D segmentation using a U-Net model.
- [config_spineGeHemis.json](../../ivadomed/config/config_spineGeHemis.json) shows how to use the HeMIS-UNet.
- [config_tumorSeg.json](../../ivadomed/config/config_tumorSeg.json) runs a 3D segmentation using a 3D UNet.

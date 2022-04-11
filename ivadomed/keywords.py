from dataclasses import dataclass


@dataclass
class ConfigKW:
    LOADER_PARAMETERS: str = "loader_parameters"
    TRAINING_PARAMETERS: str = "training_parameters"
    MODEL_NAME: str = "model_name"
    MODIFIED_3D_UNET: str = "Modified3DUNet"
    DEBUGGING: str = "debugging"
    FILMED_UNET: str = "FiLMedUnet"
    DEFAULT_MODEL: str = "default_model"
    OBJECT_DETECTION_PARAMS: str = "object_detection_params"
    GPU_IDS: str = "gpu_ids"
    PATH_OUTPUT: str = "path_output"
    POSTPROCESSING: str = "postprocessing"
    COMMAND: str = "command"
    LOG_FILE: str = "log_file"
    TRANSFORMATION: str = "transformation"
    SPLIT_DATASET: str = "split_dataset"
    UNCERTAINTY: str = "uncertainty"
    UNDO_TRANSFORMS: str = "undo_transforms"
    EVALUATION_PARAMETERS: str = "evaluation_parameters"
    HEMIS_UNET: str = "HeMISUnet"
    SPLIT_PATH: str = "split_path"
    TRAINING_SHA256: str = "training_sha256"


@dataclass
class LoaderParamsKW:
    PATH_DATA = "path_data"
    BIDS_CONFIG = "bids_config"
    TARGET_SUFFIX = "target_suffix"
    ROI_PARAMS = "roi_params"
    CONTRAST_PARAMS = "contrast_params"
    MULTICHANNEL = "multichannel"  # boolean key that is used to change the configuration file ever slightly.
    EXTENSIONS = "extensions"
    TARGET_GROUND_TRUTH = "target_ground_truth"
    TARGET_SESSIONS = "target_sessions"  # the sessions to focus the analyses on
    METADATA_TYPE = "metadata_type"
    MODEL_PARAMS = "model_params"
    SLICE_AXIS = "slice_axis"
    IS_INPUT_DROPOUT = "is_input_dropout"
    SLICE_FILTER_PARAMS = "slice_filter_params"
    SUBJECT_SELECTION = "subject_selection"
    PATCH_FILTER_PARAMS = "patch_filter_params"


@dataclass
class SplitDatasetKW:
    SPLIT_METHOD: str = "split_method"
    FNAME_SPLIT: str = "fname_split"
    DATA_TESTING: str = "data_testing"
    RANDOM_SEED: str = "random_seed"
    TRAIN_FRACTION: str = "train_fraction"
    TEST_FRACTION: str = "test_fraction"
    BALANCE: str = "balance"


@dataclass
class DataTestingKW:
    DATA_TYPE: str = "data_type"
    DATA_VALUE: str = "data_value"


@dataclass
class TrainingParamsKW:
    BALANCE_SAMPLES: str = "balance_samples"
    BATCH_SIZE: str = "batch_size"


@dataclass
class TransformationKW:
    ROICROP = "ROICrop"
    CENTERCROP = "CenterCrop"
    RESAMPLE = "Resample"
    NUMPY_TO_TENSOR = "NumpyToTensor"
    W_SPACE = "wspace"
    H_SPACE = "hspace"


@dataclass
class BalanceSamplesKW:
    APPLIED: str = "applied"
    TYPE: str = "type"


@dataclass
class ContrastParamsKW:
    CONTRAST_LST: str = "contrast_lst"  # The list help determine the number of model parameter inputs.
    BALANCE: str = "balance"
    TRAINING_VALIDATION: str = "training_validation"
    TESTING: str = "testing"


class ModelParamsKW:
    LENGTH_2D = "length_2D"
    STRIDE_2D = "stride_2D"
    LENGTH_3D = "length_3D"
    STRIDE_3D = "stride_3D"
    FILM_LAYERS = "film_layers"
    FOLDER_NAME = "folder_name"
    METADATA = "metadata"
    FILM_ONEHOTENCODER = "film_onehotencoder"
    N_METADATA = "n_metadata"
    APPLIED = "applied"
    NAME = "name"
    IS_2D = "is_2d"
    IN_CHANNEL = "in_channel"
    OUT_CHANNEL = "out_channel"
    TARGET_LST = "target_lst"
    ROI_LST = "roi_lst"
    PATH_HDF5 = "path_hdf5"
    CSV_PATH = "csv_path"
    RAM = "ram"
    ATTENTION = "attention"
    DEPTH = "depth"
    MISSING_PROBABILITY = "missing_probability"
    MISSING_PROBABILITY_GROWTH = "missing_probability_growth"
    DROPOUT_RATE = "dropout_rate"
    BN_MOMENTUM = "bn_momentum"
    FINAL_ACTIVATION = "final_activation"


@dataclass
class SubjectDictKW:
    ABSOLUTE_PATHS: str = "absolute_paths"
    DERIV_PATH: str = "deriv_path"
    ROI_FILENAME: str = "roi_filename"
    METADATA: str = "metadata"
    EXTENSIONS: str = "extensions"


@dataclass
class SubjectDataFrameKW:
    FILENAME: str = "filename"


@dataclass
class OptionKW:
    METADATA: str = "metadata"
    FNAME_PRIOR: str = 'fname_prior'
    BINARIZE_PREDICTION: str = "binarize_prediction"
    BINARIZE_MAXPOOLING: str = "binarize_maxpooling"
    KEEP_LARGEST: str = "keep_largest"
    FILL_HOLES: str = "fill_holes"
    REMOVE_SMALL: str = "remove_small"
    OVERLAP_2D: str = "overlap_2D"
    PIXEL_SIZE: str = "pixel_size"
    PIXEL_SIZE_UNITS: str = "pixel_size_units"


@dataclass
class BidsDataFrameKW:
    # bids layout converted to dataframe during bids dataset creation
    PATH: str = "path"   # full path.
    FILENAME: str = "filename"  # the actual file's name (base)
    PARTICIPANT_I: str = "participant_id"  # i.e.    sub-unf01
    SUBJECT: str = "subject"  # i.e.  unf01
    SUFFIX: str = "suffix"   # T1w
    SESSION: str = "session"  # session field (single int) in Bids DataFrame
    EXTENSION: str = "extension"   # .nii.gz
    DERIVATIVES: str = "derivatives"


@dataclass
class ROIParamsKW:
    SUFFIX: str = "suffix"
    SLICE_FILTER_ROI: str = "slice_filter_roi"


@dataclass
class MetadataKW:
    CONTRAST: str = "contrast"
    CONTRASTS: str = "contrasts"
    BOUNDING_BOX: str = "bounding_box"
    DATA_TYPE: str = "data_type"
    PRE_RESAMPLE_SHAPE: str = "preresample_shape"
    CROP_PARAMS: str = "crop_params"
    MRI_PARAMS: str = "mri_params"
    ROTATION: str = "rotation"
    TRANSLATION: str = "translation"
    SCALE: str = "scale"
    COORD: str = "coord"
    ZOOMS: str = "zooms"
    UNDO: str = "undo"
    REVERSE: str = "reverse"
    OFFSET: str = "offset"
    ELASTIC: str = "elastic"
    GAUSSIAN_NOISE: str = "gaussian_noise"
    GAMMA: str = "gamma"
    BIAS_FIELD: str = "bias_field"
    BLUR: str = "blur"
    DATA_SHAPE: str = "data_shape"
    SLICE_INDEX: str = "slice_index"
    MISSING_MOD: str = "missing_mod"
    METADATA_DICT: str = "metadata_dict"
    INDEX_SHAPE: str = "index_shape"
    GT_METADATA: str = "gt_metadata"
    GT_FILENAMES: str = "gt_filenames"
    INPUT_METADATA: str = "input_metadata"
    INPUT_FILENAMES: str = "input_filenames"
    ROI_METADATA: str = "roi_metadata"
    PIXEL_SIZE: str = "PixelSize"
    PIXEL_SIZE_UNITS: str = "PixelSizeUnits"


@dataclass
class ObjectDetectionParamsKW:
    GPU_IDS: str = "gpu_ids"
    PATH_OUTPUT: str = "path_output"
    OBJECT_DETECTION_PATH: str = "object_detection_path"
    SAFETY_FACTOR: str = "safety_factor"


@dataclass
class UncertaintyKW:
    ALEATORIC: str = 'aleatoric'
    N_IT: str = "n_it"


@dataclass
class PostprocessingKW:
    BINARIZE_PREDICTION: str = "binarize_prediction"


@dataclass
class BinarizeProdictionKW:
    THR: str = "thr"


@dataclass
class SliceFilterParamsKW:
    FILTER_EMPTY_MASK: str = "filter_empty_mask"


@dataclass
class IgnoredFolderKW:
    MACOSX: str = "__MACOSX"


@dataclass
class MetricsKW:
    RECALL_SPECIFICITY: str = "recall_specificity"
    DICE: str = "dice"

@dataclass
class MetadataParamsKW:
    CONTRAST = "contrast"
    BOUNDING_BOX = "bounding_box"

@dataclass
class MetadataChoiceKW:
    MRI_PARAMS = "mri_params"
    CONTRASTS = "contrasts"
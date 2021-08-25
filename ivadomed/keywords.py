from dataclasses import dataclass


@dataclass
class ConfigKW:
    LOADER_PARAMETERS = "loader_parameters"
    TRAINING_PARAMETERS = "training_parameters"
    MODEL_NAME = "model_name"
    MODIFIED3DUNET = "Modified3DUNet"
    DEBUGGING = "debugging"
    FILMEDUNET = "FiLMedUnet"
    DEFAULT_MODEL = "default_model"
    OBJECT_DETECTION_PARAMS = "object_detection_params"
    GPU_IDS = "gpu_ids"
    PATH_OUTPUT = "path_output"
    POSTPROCESSING = "postprocessing"
    COMMAND = "command"
    LOG_FILE = "log_file"
    TRANSFORMATION = "transformation"
    SPLIT_DATASET = "split_dataset"
    UNCERTAINTY = "uncertainty"
    UNDO_TRANSFORMS = "undo_transforms"
    ELEVATION_PARAMETERS = "elevation_parameters"
    HEMISUNET = "HeMISUnet"
    SPLIT_PATH = "split_path"
    TRAINING_SHA256 = "training_sha256"


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


@dataclass
class SplitDatasetKW:
    SPLIT_METHOD = "split_method"
    FNAME_SPLIT = "fname_split"
    DATA_TESTING = "data_testing"
    RANDOM_SEED = "random_seed"
    TRAIN_FRACTION = "train_fraction"
    TEST_FRACTION = "test_fraction"
    BALANCE = "balance"


@dataclass
class DataTestingKW:
    DATA_TYPE = "data_type"
    DATA_VALUE = "data_value"


@dataclass
class TrainingParamsKW:
    BALANCE_SAMPLES = "balance_samples"
    BATCH_SIZE = "batch_size"


@dataclass
class TransformationKW:
    ROICROP = "ROICrop"
    CENTERCROP = "CenterCrop"
    RESAMPLE = "Resample"


@dataclass
class BalanceSamplesKW:
    APPLIED = "applied"
    TYPE = "type"


@dataclass
class ContrastParamsKW:
    CONTRAST_LIST = "contrast_lst"  # The list help determine the number of model parameter inputs.
    BALANCE = "balance"
    TRAINING_VALIDATION = "training_validation"
    TESTING = "testing"


@dataclass
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
    TARGET_LIST = "target_list"
    ROI_LIST = "roi_list"
    PATH_HDF5 = "path_hdf5"
    CSV_PATH = "csv_path"
    RAM = "ram"
    ATTENTION = "attention"
    DEPTH = "depth"
    MISSING_PROBABILITY = "missing_probability"
    MISSING_PROBABILITY_GROWTH = "missing_probability_growth"


@dataclass
class SubjectDictKW:
    ABSOLUTE_PATHS = "absolute_paths"
    DERIV_PATH = "deriv_path"
    ROI_FILENAME = "roi_filename"
    METADATA = "metadata"
    EXTENSIONS = "extensions"


@dataclass
class SubjectDataFrameKW:
    FILENAME = "filename"


@dataclass
class BidsDataFrameKW:
    # bids layout converted to dataframe during bids dataset creation
    PATH = "path"   # full path.
    FILENAME = "filename"  # the actual file's name (base)
    PARTICIPANT_ID = "participant_id"  # i.e.    sub-unf01
    SUBJECT = "subject"  # i.e.  unf01
    SUFFIX = "suffix"   # T1w
    SESSION = "session"  # session field (single int) in Bids DataFrame
    EXTENSION = "extension"   # .nii.gz
    DERIVATIVES = "derivatives"


@dataclass
class ROIParamsKW:
    SUFFIX = "suffix"
    SLICE_FILTER_ROI = "slice_filter_roi"


@dataclass
class MetadataParamsKW:
    CONTRAST = "contrast"
    BOUNDING_BOX = "bounding_box"


@dataclass
class MetadataChoiceKW:
    MRI_PARAMS = "mri_params"
    CONTRASTS = "contrasts"


@dataclass
class ObjectDetectionParamsKW:
    GPU_IDS = "gpu_ids"
    PATH_OUTPUT = "path_output"
    OBJECT_DETECTION_PATH = "object_detection_path"
    SAFETY_FACTOR = "safety_factor"


@dataclass
class UncertaintyKW:
    ALEATORIC = 'aleatoric'
    N_IT = "n_it"


@dataclass
class PostprocessingKW:
    BINARIZE_PREDICTION = "binarize_prediction"


@dataclass
class BinarizeProdictionKW:
    THR = "thr"


@dataclass
class SliceFilterParamsKW:
    FILTER_EMPTY_MASK = "filter_empty_mask"


@dataclass
class IgnoredFolderKW:
    MACOSX = "__MACOSX"
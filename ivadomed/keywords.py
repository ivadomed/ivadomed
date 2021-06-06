from dataclasses import dataclass


@dataclass
class LoaderParamsKW:
    PATH_DATA = "path_data"
    BIDS_CONFIG = "bids_config"
    TARGET_SUFFIX = "target_suffix"
    ROI_PARAMS = "roi_params"
    CONTRAST_PARAMS = "contrast_params"
    MULTICHANNEL = "multichannel"
    EXTENSIONS = "extensions"


@dataclass
class ContrastParamsKW:
    CONTRAST_LIST = "contrast_lst"
    BALANCE = "balance"


@dataclass
class ModelParamsKW:
    LENGTH_2D = "length_2D"
    STRIDE_2D = "stride_2D"



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
    FILENAME = "filename"
    PARTICIPANT_ID = "participant_id"
    SUFFIX = "suffix"
    PATH = "path"


@dataclass
class ROIParamsKW:
    SUFFIX = "suffix"


@dataclass
class MetadataParamsKW:
    CONTRAST = "contrast"
    BOUNDING_BOX = "bounding_box"


@dataclass
class MetadataChoiceKW:
    MRI_PARAMS = "mri_params"
    CONTRASTS = "contrasts"


@dataclass
class FileStemKW:
    """
    Everything MUST be using lower case here to allow case insensitive file name comparisons.
    """
    list_subject = ["sub-", "subject", "subj"]
    list_acquisition = ["acq-"]
    list_sample = ["sample-"]
    list_session = ["ses-", "session-"]
    list_run = ["run-"]
    list_contrasts = ["t1w", "t2w", "t2star", "ct", "sem"]
    list_derivative = ["lesion-", "seg-", "labels-"]
    list_location = ["mid"]
    list_stats = ["heatmap"]


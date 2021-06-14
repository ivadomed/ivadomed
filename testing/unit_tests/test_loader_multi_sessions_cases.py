from testing.common_testing_util import path_data_multi_sessions_contrasts_source
from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW, ROIParamsKW


def case_data_multi_session_contrast():
    loader_parameters = {
        LoaderParamsKW.PATH_DATA: [path_data_multi_sessions_contrasts_source],
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1"],
        LoaderParamsKW.EXTENSIONS: [".nii", ".nii.gz"],
        "roi_params": {"suffix": None, "slice_filter_roi": None},
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["t1w", "t2w", "flair", "pd"]
        }
    }

    return loader_parameters

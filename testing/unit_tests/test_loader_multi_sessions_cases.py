from testing.common_testing_util import path_data_multi_sessions_contrasts_tmp
from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW, ROIParamsKW


def case_data_multi_session_contrast():
    loader_parameters = {
        LoaderParamsKW.MULTICHANNEL: "true",
        LoaderParamsKW.TARGET_SESSIONS: [1, 2, 3, 4],
        LoaderParamsKW.TARGET_GROUND_TRUTH: "_lesion-manual-rater1",
        LoaderParamsKW.PATH_DATA: [path_data_multi_sessions_contrasts_tmp],
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1", "_lesion-manual-rater2"],
        LoaderParamsKW.EXTENSIONS: [".nii", ".nii.gz"],
        "roi_params": {"suffix": None, "slice_filter_roi": None},
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T1w", "T2w", "FLAIR", "PD"]
        }
    }

    return loader_parameters, "df_ref.csv"


def case_data_target_specific_session_contrast():
    loader_parameters = {
        LoaderParamsKW.MULTICHANNEL: "true",
        LoaderParamsKW.TARGET_SESSIONS: [2,3],
        LoaderParamsKW.TARGET_GROUND_TRUTH: "_lesion-manual-rater1",
        LoaderParamsKW.PATH_DATA: [path_data_multi_sessions_contrasts_tmp],
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1", "_lesion-manual-rater2"],
        LoaderParamsKW.EXTENSIONS: [".nii", ".nii.gz"],
        "roi_params": {"suffix": None, "slice_filter_roi": None},
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T1w", "T2w", "FLAIR", "PD"]
        }
    }

    return loader_parameters, "df_ref_selective_session.csv"


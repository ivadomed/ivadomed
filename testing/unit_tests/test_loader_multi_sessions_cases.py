from testing.common_testing_util import path_data_multi_sessions_contrasts_tmp
from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW

# A default dict which subsequent tests attempt to deviate from
default_loader_parameters: dict = {
    LoaderParamsKW.MULTICHANNEL: "true",
    LoaderParamsKW.TARGET_SESSIONS: ["01", "02", "03", "04"],
    LoaderParamsKW.PATH_DATA: [path_data_multi_sessions_contrasts_tmp],
    LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1", "_lesion-manual-rater2"],
    LoaderParamsKW.EXTENSIONS: [".nii", ".nii.gz"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    LoaderParamsKW.CONTRAST_PARAMS: {
        ContrastParamsKW.CONTRAST_LIST: ["T1w", "T2w", "FLAIR", "PD"]
    }
}
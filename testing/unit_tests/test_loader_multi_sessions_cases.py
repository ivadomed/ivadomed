from testing.common_testing_util import path_data_multi_sessions_contrasts_tmp
from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW
from copy import deepcopy

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


def case_data_multi_session_contrast():
    loader_parameters = deepcopy(default_loader_parameters)

    return loader_parameters, "df_ref.csv"


def case_data_multi_session_contrast_missing_session():
    loader_parameters = deepcopy(default_loader_parameters)
    # Variation of missing session file is introduced at the TEST level, not at the case level.
    return loader_parameters, "df_ref_missing_session.csv"


def case_data_multi_session_contrast_mismatching_target_suffix():
    loader_parameters = deepcopy(default_loader_parameters)

    # Update adding extraneous target suffix that doesn't exist.
    loader_parameters.update(
        {LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1", "_lesion-manual-rater2", "aa"]}
    )

    return loader_parameters, "df_ref_missing_session.csv"


def case_data_multi_session_contrast_missing_modality():
    loader_parameters = deepcopy(default_loader_parameters)

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T1w", "T2w"]
        }
    })

    return loader_parameters, "df_ref_missing_modality.csv"


def case_data_target_specific_session_contrast():
    loader_parameters = deepcopy(default_loader_parameters)

    # Target reduced sessions
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["02", "03"],
    })

    return loader_parameters, "df_ref_selective_session.csv"


def case_data_target_single_subject_with_session():
    loader_parameters = deepcopy(default_loader_parameters)

    # Target non-existent target session
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["05"]
    })

    return loader_parameters, "df_ref_single_session.csv"

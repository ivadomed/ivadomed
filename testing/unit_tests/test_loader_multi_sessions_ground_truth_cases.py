from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW
from copy import deepcopy
from test_loader_multi_sessions_cases import default_loader_parameters

# Ground Truth

def case_more_ground_truth_than_available():
    loader_parameters = deepcopy(default_loader_parameters)

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1", "_lesion-manual-rater2", "_lesion_manual-rater3"],
        }
    })

    return loader_parameters, "df_ref_missing_modality.csv"


def case_less_ground_truth_than_available():
    loader_parameters = deepcopy(default_loader_parameters)

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1"],
        }
    })

    return loader_parameters, "df_ref_missing_modality.csv"


def case_missing_ground_truth():
    loader_parameters = deepcopy(default_loader_parameters)

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater3"],
        }
    })

    return loader_parameters, "df_ref_missing_modality.csv"

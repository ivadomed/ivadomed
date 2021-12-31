from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW
from copy import deepcopy
from testing.unit_tests.test_loader_multi_sessions_cases import default_loader_parameters


# Modality Related Testing

def case_more_contrasts_than_available():
    loader_parameters = deepcopy(default_loader_parameters)

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T1w", "T2w", "FLAIR", "PD", "CT"]
        }
    })

    return loader_parameters, "df_ref_missing_modality.csv"


def case_partially_available_contrasts():
    loader_parameters = deepcopy(default_loader_parameters)

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T1w", "T2w", "CT"]
        }
    })

    return loader_parameters, "df_ref_missing_modality.csv"


def case_less_contrasts_than_available():
    loader_parameters = deepcopy(default_loader_parameters)

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T1w", "T2w"]
        }
    })

    return loader_parameters, "df_ref_missing_modality.csv"


def case_single_contrast():
    loader_parameters = deepcopy(default_loader_parameters)

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T1w"]
        }
    })

    return loader_parameters, "df_ref_missing_modality.csv"


def case_unavailable_contrast():
    loader_parameters = deepcopy(default_loader_parameters)

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["CT"]
        }
    })

    return loader_parameters, "df_ref_missing_modality.csv"


def case_not_specified_contrast():
    loader_parameters = deepcopy(default_loader_parameters)

    # Reduce MRI contrast list
    loader_parameters.pop(LoaderParamsKW.CONTRAST_PARAMS)

    return loader_parameters, "df_ref_missing_modality.csv"

from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW
from testing.common_testing_util import get_multi_default_case


# Ground Truth

def case_more_ground_truth_than_available():
    loader_parameters = get_multi_default_case()

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1", "_lesion-manual-rater2", "_lesion_manual-rater3"],
    })

    return loader_parameters, "df_ref_more_target_suffix.csv"

def case_partially_available_ground_truth():
    loader_parameters = get_multi_default_case()

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1", "_lesion_manual-rater3"],
    })

    return loader_parameters, "df_ref_fewer_target_suffix.csv"

def case_less_ground_truth_than_available():
    loader_parameters = get_multi_default_case()

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1"],
    })

    return loader_parameters, "df_ref_fewer_target_suffix.csv"


def case_unavailable_ground_truth():
    # Target suffix does not EXIST!
    # Therefore
    loader_parameters = get_multi_default_case()

    # Reduce MRI contrast list
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater3"],
    })

    return loader_parameters

from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW
from testing.common_testing_util import get_multi_default_case


# Ground truth related testing scenarios

def case_more_ground_truth_than_available():
    """Test scenario where some user specified more ground truth to index for ivadomed than what exists at the file
    level"""

    # Get default multi-ground truth loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # All existing files plus "_lesion_manual-rater3", which does not exist at the file level
    # This should result in a CSV equivalent to the same as asking for only rater1&2
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1", "_lesion-manual-rater2", "_lesion_manual-rater3"],
    })

    return loader_parameters, "df_ref_more_target_suffix.csv"


def case_partially_available_ground_truth():
    """Test scenario where some user specified more ground truth are available to be indexed for ivadomed than what
    exists at the file level"""

    # Get default multi-ground truth loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Existing "_lesion_manual-rater1" files plus "_lesion_manual-rater3", which does not exist at the file level
    # This should result in a CSV equivalent to the same as asking for only rater1
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1", "_lesion_manual-rater3"],
    })

    return loader_parameters, "df_ref_fewer_target_suffix.csv"


def case_less_ground_truth_than_available():
    """Test scenario where some user specified less ground truth than what is available to be indexed for ivadomed"""

    # Get default multi-ground truth loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Index only "_lesion_manual-rater1" and see how it handles other files that exist at the file level
    # This should result in a CSV equivalent to the same as asking for only rater1
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1"],
    })

    return loader_parameters, "df_ref_fewer_target_suffix.csv"


def case_unavailable_ground_truth():
    """Test scenario where ground truth doesn't exist"""

    # Get default multi-ground truth loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Indexing for "_lesion_manual-rater3", which does not exist at the file level
    # This should raise a exception, when ground truth does not exist.
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater3"],
    })

    return loader_parameters


def case_not_specified_ground_truth():
    """Test scenario where the if user does not specify the ground truth information."""

    # Get default multi-session loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Remove ground truth specification
    # This shoudl raise exception
    loader_parameters.pop(LoaderParamsKW.TARGET_SUFFIX)

    return loader_parameters

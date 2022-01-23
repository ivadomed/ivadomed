from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW
from testing.common_testing_util import get_multi_default_case

# Modality/MRI Contrasts related testing scenarios


def case_more_contrasts_than_available():
    """Test Scenario where some user specified imaging contrasts to index for ivadomed that does not exist at the file
    level while other contrasts exist. """

    # Get default multi-contrast loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # All existing files plus CT, which does not exist at the file level
    # This should result in a EMPTY data frame
    # No return CSV to compare with.
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T1w", "T2w", "FLAIR", "PD", "CT"]
        }
    })

    return loader_parameters


def case_partially_available_contrasts():
    """Test Scenario where the user specified an imaging contrast to index for ivadomed that does not exist at the file
        level while other contrasts exist. """

    # Get default multi-contrast loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Reduce MRI contrast list while adding CT, which does not exist at the file level
    # This should result in a EMPTY data frame
    # No return CSV to compare with.
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T1w", "T2w", "CT"]
        }
    })

    return loader_parameters


def case_less_contrasts_than_available():
    """Test scenario where a few contrasts parameter are provided. Fairly typical scenario"""

    # Get default multi-contrast loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Reduce MRI contrast list to PD/FLAIR only. Simulating TWO contrast scenario when MULTIPLE imaging contrasts files
    # are available (T1, T2, Flair, PD)
    # This should result in a fairly normal csv as part fo the return.
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["PD", "FLAIR"]
        }
    })

    return loader_parameters, "df_ref_flair_pd.csv"


def case_single_contrast():
    """Test scenario where a single contrast parameter is provided. Fairly typical scenario"""

    # Get default multi-contrast loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Reduce MRI contrast list to T2W only. Simulating SINGLE contrast scenario when MULTIPLE imaging contrasts files
    # are available
    # This should result in a fairly normal csv as part fo the return.
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T2w"]
        }
    })

    return loader_parameters, "df_ref_missing_modality.csv"


def case_unavailable_contrast():
    """Test Scenario where the user specified a imaging contrast to index for ivaodmed that does not exist at the file
    level. """

    # Get default multi-contrast loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Update the contrast parameter to overwrite it with an imaging contrast which does not exist at the file level
    # This should result in ValueError exception asking user to review loader configuration JSON.
    # No return CSV to compare with.
    loader_parameters.update({
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["CT"]
        }
    })

    return loader_parameters


def case_not_specified_contrast():
    """Test Scenario where the user did not provide any imaging contrast to index for ivaodmed. At least one contrast is
    usually required"""

    # Get default multi-contrast loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Remove MRI contrast list entirely.
    # This should result in an RunTime exception at the unit test level (as at least one contrast parameter is needed)
    # No return CSV to compare with.
    loader_parameters.pop(LoaderParamsKW.CONTRAST_PARAMS)

    return loader_parameters

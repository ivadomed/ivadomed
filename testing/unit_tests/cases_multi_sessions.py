from testing.common_testing_util import path_data_multi_sessions_contrasts_tmp, get_multi_default_case
from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW
from copy import deepcopy


# Session Related Testing
def case_sessions_only_subjects_have():
    """Since only ms03 has session 05, only ms03 should be included."""
    loader_parameters = get_multi_default_case()
    # Target more sessions than what we have data for
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["01", "02", "03", "04", "05"],
    })

    return loader_parameters, "df_ref_subject_3_only.csv"

def case_more_sessions_than_available():
    """Since NO ONE has session 06, no one should be included. An empty data frame should be generated. """
    loader_parameters = get_multi_default_case()
    # Target more sessions than what we have data for
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["01", "02", "03", "04", "05", "06"],
    })

    return loader_parameters

def case_partially_available_sessions():
    """Since NO ONE has session 06, no one should be included. An EMPTY dataframe should be generated.
    even if we ask fewer sessions"""
    loader_parameters = get_multi_default_case()
    # Target less sessions than we have data for
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["04", "05", "06"],
    })

    return loader_parameters


def case_less_sessions_than_available():
    """Since EVERYONE has session 02 and 03, EVERYONE should be included.
    We also include ground truth but not data from outside of those sessions as well."""
    loader_parameters = get_multi_default_case()
    # Target less sessions than we have data for
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["02", "03"],
    })

    return loader_parameters, "df_ref_selective_sessions.csv"


def case_single_session():
    """Since only ms03 has session 05, only ms03 should be included."""
    loader_parameters = get_multi_default_case()

    # Target mostly non-existent target session
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["05"]
    })

    return loader_parameters, "df_ref_session_5_only.csv"


def case_unavailable_session():
    """Since NO ONE has session 06, no one should be included."""
    loader_parameters = get_multi_default_case()

    # Target non-existent target session
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["06"]
    })

    return loader_parameters


def case_not_specified_session():
    """This should in theory throw an exception???"""
    loader_parameters = get_multi_default_case()

    # Assume no session
    loader_parameters.pop(LoaderParamsKW.TARGET_SESSIONS)

    return loader_parameters, "df_ref.csv"

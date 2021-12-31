from testing.common_testing_util import path_data_multi_sessions_contrasts_tmp
from ivadomed.keywords import LoaderParamsKW, ContrastParamsKW
from copy import deepcopy
from test_loader_multi_sessions_cases import default_loader_parameters


# Session Related Testing
def case_more_sessions_than_available():
    loader_parameters = deepcopy(default_loader_parameters)
    # Target more sessions than what we have data for
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["01", "02", "03", "04", "05"],
    })

    return loader_parameters, "df_ref_selective_session.csv"


def case_partially_available_sessions():
    loader_parameters = deepcopy(default_loader_parameters)
    # Target less sessions than we have data for
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["01", "02", "03", "04", "05", "06"],
    })

    return loader_parameters, "df_ref_selective_session.csv"


def case_less_sessions_than_available():
    loader_parameters = deepcopy(default_loader_parameters)
    # Target less sessions than we have data for
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["02", "03"],
    })

    return loader_parameters, "df_ref_selective_session.csv"


def case_single_session():
    loader_parameters = deepcopy(default_loader_parameters)

    # Target mostly non-existent target session
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["05"]
    })

    return loader_parameters, "df_ref_single_session.csv"


def case_unavailable_session():
    loader_parameters = deepcopy(default_loader_parameters)

    # Target non-existent target session
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["06"]
    })

    return loader_parameters, "df_ref_single_session.csv"


def case_not_specified_session():
    loader_parameters = deepcopy(default_loader_parameters)

    # Assume no session
    loader_parameters.pop(LoaderParamsKW.TARGET_SESSIONS)

    return loader_parameters, "df_ref_single_session.csv"

from testing.common_testing_util import get_multi_default_case
from ivadomed.keywords import LoaderParamsKW

# Multi sessions related testing scenarios


def case_sessions_only_subjects_have():
    """Test Scenario where user specified sessions to index for ivadomed that contain files
    does not exist at the file level for ALL subjects while other contrasts exist for all subjects.
    """

    # Get default multi-session loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Target more sessions than what we have data for:
    # Since only ms03 has session 05, only ms03 should be included as all specified sessions are
    # mandatory required.
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["01", "02", "03", "04", "05"],
    })

    return loader_parameters, "df_ref_subject_3_only.csv"


def case_more_sessions_than_available():
    """Test Scenario where user specified sessions to index for ivadomed that contain session
    which does not exist for any subjects.
    """

    # Get default multi-session loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Since NO ONE has session 06, no one should be included. An empty data frame should be generated.
    # Target more sessions than what we have data for
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["01", "02", "03", "04", "05", "06"],
    })

    return loader_parameters


def case_partially_available_sessions():
    """Test Scenario where user specified some but not all of available sessions to index for ivadomed that contain
    session which does not exist for any subjects.
    """

    # Get default multi-session loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Since NO ONE has session 06, no one should be included. An EMPTY dataframe should be generated.
    # even if we ask fewer sessions
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["04", "05", "06"],
    })

    return loader_parameters


def case_less_sessions_than_available():
    """Test Scenario where user specified some but not all of available sessions to index for ivadomed
    """

    # Get default multi-session loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Target less sessions than what is available at the file level.
    # Since EVERYONE has session 02 and 03, EVERYONE should be included.
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["02", "03"],
    })

    return loader_parameters, "df_ref_selective_sessions.csv"


def case_single_session():
    """Test scenario where user specified session only exist for a single subject."""

    # Get default multi-session loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Target mostly non-existent target session
    # Since only ms03 has session 05, only ms03 should be included
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["05"]
    })

    return loader_parameters, "df_ref_session_5_only.csv"


def case_unavailable_session():
    """Test scenario where the no one has the session. No one should be included."""

    # Get default multi-session loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Target non-existent target session
    # No valid session should raise an exception
    loader_parameters.update({
        LoaderParamsKW.TARGET_SESSIONS: ["06"]
    })

    return loader_parameters


def case_not_specified_session():
    """Test scenario where the if user does not specify the session information.
    When that happens, we assume we are not picky about the session information and therefore,
    any sessions can be used."""

    # Get default multi-session loader parameter to be overwritten
    loader_parameters = get_multi_default_case()

    # Remove session specification, should include all possible sessions data
    loader_parameters.pop(LoaderParamsKW.TARGET_SESSIONS)

    return loader_parameters, "df_ref.csv"

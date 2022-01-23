import os
import shutil

from ivadomed.loader.bids_dataframe import BidsDataframe
from testing.unit_tests.t_utils import path_temp, download_multi_data
from testing.common_testing_util import remove_tmp_dir, path_data_multi_sessions_contrasts_source, \
    assert_empty_bids_dataframe, bids_dataframe_comparison_framework, path_data_multi_sessions_contrasts_tmp
from pytest_cases import parametrize_with_cases
import pytest
from testing.unit_tests.cases_multi_sessions import *
from testing.unit_tests.cases_multi_contrasts import *
from testing.unit_tests.cases_multi_ground_truth import *


def setup_function():
    # Dedicated setup function for multi-session data.
    remove_tmp_dir()
    os.mkdir(path_temp)
    if os.path.exists(path_data_multi_sessions_contrasts_source):
        shutil.copytree(path_data_multi_sessions_contrasts_source,
                        path_data_multi_sessions_contrasts_tmp,
                        ignore=shutil.ignore_patterns(str(path_data_multi_sessions_contrasts_source / '.git')))


@parametrize_with_cases("loader_parameters, target_csv", cases=[
    case_sessions_only_subjects_have,
    case_less_sessions_than_available,
    case_single_session,
    case_not_specified_session,  # default to accept ALL possible sessions
])
def test_valid_multi_sessions(download_multi_data,
                              loader_parameters,
                              target_csv):
    """
    Test for when multi-sessions and multi-contrasts, how the filtering and ground truth identification process works.
    """
    bids_dataframe_comparison_framework(loader_parameters, target_csv)


@parametrize_with_cases("loader_parameters", cases=[
    case_more_sessions_than_available,
    case_partially_available_sessions,
    case_more_contrasts_than_available,
    case_partially_available_contrasts,
])
def test_invalid_empty_dataframes(download_multi_data,
                                  loader_parameters):
    assert_empty_bids_dataframe(loader_parameters)


@parametrize_with_cases("loader_parameters", cases=[
    case_not_specified_contrast,  # Contrast specification is required.
])
def test_raise_value_errors(download_multi_data,
                            loader_parameters):
    with pytest.raises(ValueError):
        BidsDataframe(loader_parameters,
                      str(path_data_multi_sessions_contrasts_tmp),
                      derivatives=True)


@parametrize_with_cases("loader_parameters", cases=[
    case_unavailable_session,
    case_unavailable_contrast,
    case_unavailable_ground_truth,
])
def test_raise_runtime_errors(download_multi_data,
                              loader_parameters):
    """
    Test run time error which are raised when Step 2 filter in BIDS loader failed to have any files remaining.
    """
    with pytest.raises(RuntimeError):
        BidsDataframe(loader_parameters,
                      str(path_data_multi_sessions_contrasts_tmp),
                      derivatives=True)


@parametrize_with_cases("loader_parameters, target_csv", cases=[
    case_less_contrasts_than_available,
    case_single_contrast,
])
def test_valid_multi_contrasts(
        download_multi_data,
        loader_parameters,
        target_csv):
    """
    Test for multi-contrasts, how the filtering and ground truth identification process works.
    """
    bids_dataframe_comparison_framework(loader_parameters, target_csv)


@parametrize_with_cases("loader_parameters, target_csv", cases=[
    case_more_ground_truth_than_available,
    case_less_ground_truth_than_available,
    case_partially_available_ground_truth
])
def test_valid_multi_target_suffixes(
        download_multi_data,
        loader_parameters,
        target_csv):
    """
    Test for target suffixes
    """
    bids_dataframe_comparison_framework(loader_parameters, target_csv)


def teardown_function():
    remove_tmp_dir()
    pass

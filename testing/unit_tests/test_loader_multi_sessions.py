import os
import csv_diff
import shutil

from ivadomed.loader.bids_dataframe import BidsDataframe
from testing.unit_tests.t_utils import path_temp, download_data_multi_sessions_contrasts_test_files
from testing.common_testing_util import remove_tmp_dir, path_data_multi_sessions_contrasts_source, \
    path_data_multi_sessions_contrasts_tmp
from pytest_cases import parametrize_with_cases

from testing.unit_tests.test_loader_multi_sessions_session_cases import *
from testing.unit_tests.test_loader_multi_sessions_contrasts_cases import *
from testing.unit_tests.test_loader_multi_sessions_ground_truth_cases import *
from loguru import logger
from ivadomed.keywords import BidsDataFrameKW, LoaderParamsKW

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


def setup_function():
    # Dedicated setup function for multi-session data.
    remove_tmp_dir()
    os.mkdir(path_temp)
    if os.path.exists(path_data_multi_sessions_contrasts_source):
        shutil.copytree(path_data_multi_sessions_contrasts_source,
                        path_data_multi_sessions_contrasts_tmp,
                        ignore=shutil.ignore_patterns(str(path_data_multi_sessions_contrasts_source / '.git')))


def bids_dataframe_comparison_framework(loader_parameters: dict, target_csv: str):
    """
    Main test function used to setup a CSV comparison framework between expected vs the output from the test
    Args:
        loader_parameters: dict
        target_csv:

    Returns:

    """
    # Create the bids frame.
    bids_df = BidsDataframe(loader_parameters,
                            str(path_data_multi_sessions_contrasts_tmp),
                            derivatives=True)
    # Drop path as that can varies across runs.
    df_test = bids_df.df.drop(columns=[BidsDataFrameKW.PATH])

    # Sorting to ensure consistencies.
    df_test = df_test.sort_values(by=[BidsDataFrameKW.FILENAME]).reset_index(drop=True)

    # Compare the output with the target reference CSV from the data repo.
    csv_ref = os.path.join(loader_parameters[LoaderParamsKW.PATH_DATA][0], target_csv)

    csv_test = os.path.join(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)

    # Calculate differences and ensure they are the same.
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [],
                    'columns_added': [], 'columns_removed': []}


@parametrize_with_cases("loader_parameters, target_csv", cases=[
    case_more_sessions_than_available,
    case_less_sessions_than_available,
    case_partially_available_sessions,
    case_single_session,
    case_unavailable_session,
    case_not_specified_session,
])
def test_multi_sessions_dataframe(download_data_multi_sessions_contrasts_test_files,
                                  loader_parameters,
                                  target_csv):
    """
    Test for when multi-sessions and multi-contrasts, how the filtering and ground truth identification process works.
    """
    bids_dataframe_comparison_framework(loader_parameters, target_csv)


@parametrize_with_cases("loader_parameters, target_csv", cases=[
    case_more_contrasts_than_available,
    case_less_contrasts_than_available,
    case_partially_available_contrasts,
    case_single_contrast,
    case_unavailable_contrast,
    case_not_specified_contrast,
])
def test_multi_contrasts_dataframe(
        download_data_multi_sessions_contrasts_test_files,
        loader_parameters,
        target_csv):
    """
    Test for multi-contrasts, how the filtering and ground truth identification process works.
    """
    bids_dataframe_comparison_framework(loader_parameters, target_csv)


@parametrize_with_cases("loader_parameters, target_csv", cases=[
    case_more_ground_truth_than_available,
    case_less_ground_truth_than_available,
    case_missing_ground_truth,
])
def test_multi_target_suffix(
        download_data_multi_sessions_contrasts_test_files,
        loader_parameters,
        target_csv):
    """
    Test for when derivative target suffix mismatches
    """
    bids_dataframe_comparison_framework(loader_parameters, target_csv)


def teardown_function():
    remove_tmp_dir()
    pass

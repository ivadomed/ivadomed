import os
import pytest
import csv_diff
import torch
import ivadomed.loader.tools.bids_dataframe
import ivadomed.loader.tools.utils
from testing.unit_tests.t_utils import create_tmp_dir, path_repo_root, path_temp, \
    download_data_testing_test_files, download_data_multi_sessions_contrasts_test_files
from testing.common_testing_util import remove_tmp_dir, path_data_multi_sessions_contrasts_source, path_data_multi_sessions_contrasts_tmp
from ivadomed.loader import loader as imed_loader
from loguru import logger
from pytest_cases import parametrize_with_cases
import shutil
from testing.unit_tests.test_loader_multi_sessions_cases import case_data_multi_session_contrast


def setup_function():
    # Dedicated setup function for multisession data.
    remove_tmp_dir()
    os.mkdir(path_temp)
    if os.path.exists(path_data_multi_sessions_contrasts_source):
        shutil.copytree(path_data_multi_sessions_contrasts_source,
                        path_data_multi_sessions_contrasts_tmp)


@parametrize_with_cases("loader_parameters", cases=case_data_multi_session_contrast)
def test_bids_multi_session_contrast_df_anat(download_data_multi_sessions_contrasts_test_files, loader_parameters):
    """
    Test for MRI anat nii.gz file format
    Test for when no file extensions are provided
    Test for multiple target_suffix
    Test behavior when "roi_suffix" is not None
    """

    bids_df = ivadomed.loader.tools.bids_dataframe.BidsDataframe(loader_parameters,
                                                                 str(path_data_multi_sessions_contrasts_tmp),
                                                                 derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = os.path.join(loader_parameters["path_data"][0], "df_ref.csv")
    csv_test = os.path.join(loader_parameters["path_data"][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [],
                    'columns_added': [], 'columns_removed': []}


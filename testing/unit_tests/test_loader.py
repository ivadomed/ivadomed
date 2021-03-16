import os
import pytest
import csv_diff
import logging
from unit_tests.t_utils import remove_tmp_dir, create_tmp_dir, __data_testing_dir__, __tmp_dir__
from ivadomed.loader import utils as imed_loader_utils
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [os.path.join(__data_testing_dir__, "microscopy_png")],
    "bids_config": "ivadomed/config/config_bids.json",
    "target_suffix": [["_seg-myelin-manual", "_seg-axon-manual"]],
    "extensions": [".png"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": []}
    }])
def test_bids_df_microscopy_png(loader_parameters):
    """
    Test for microscopy png file format
    Test for _sessions.tsv and _scans.tsv files
    Test for target_suffix as a nested list
    Test for when no contrast_params are provided
    """

    bids_df = imed_loader_utils.BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = os.path.join(loader_parameters["path_data"][0], "df_ref.csv")
    csv_test = os.path.join(loader_parameters["path_data"][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [], 'columns_added': [], 'columns_removed': []}


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [__data_testing_dir__],
    "target_suffix": ["_lesion-manual"],
    "extensions": [],
    "roi_params": {"suffix": "_seg-manual", "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": ["T1w", "T2w"]}
    }])
def test_bids_df_anat(loader_parameters):
    """
    Test for MRI anat nii.gz file format
    Test for when no file extensions are provided
    Test for multiple target_suffix
    Test behavior when "roi_suffix" is not None
    """

    bids_df = imed_loader_utils.BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = os.path.join(loader_parameters["path_data"][0], "df_ref.csv")
    csv_test = os.path.join(loader_parameters["path_data"][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [],
                    'columns_added': [], 'columns_removed': []}


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [__data_testing_dir__, os.path.join(__data_testing_dir__, "microscopy_png")],
    "bids_config": "ivadomed/config/config_bids.json",
    "target_suffix": ["_seg-manual", "seg-axon-manual"],
    "extensions": [".nii.gz", ".png"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": ["T1w", "T2w", "SEM"]}
    }])
def test_bids_df_multi(loader_parameters):
    """
    Test for multiple folders in path_data
    """

    bids_df = imed_loader_utils.BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = os.path.join(loader_parameters["path_data"][0], "df_ref_multi.csv")
    csv_test = os.path.join(loader_parameters["path_data"][0], "df_test_multi.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [],
                    'columns_added': [], 'columns_removed': []}


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [os.path.join(__data_testing_dir__, "ct_scan")],
    "bids_config": "ivadomed/config/config_bids.json",
    "target_suffix": ["_seg-manual"],
    "extensions": [".nii.gz"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": ["ct"]}
    }])
def test_bids_df_ctscan(loader_parameters):
    """
    Test for ct-scan
    Test for when dataset_description.json is not present in derivatives folder
    """

    bids_df = imed_loader_utils.BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = os.path.join(loader_parameters["path_data"][0], "df_ref.csv")
    csv_test = os.path.join(loader_parameters["path_data"][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [], 'columns_added': [], 'columns_removed': []}


def teardown_function():
    remove_tmp_dir()

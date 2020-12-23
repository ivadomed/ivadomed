import os

import pytest
import pandas as pd
import csv_diff

from ivadomed.loader import utils as imed_loader_utils


@pytest.mark.parametrize('loader_parameters', [{
    "bids_path": "testing_data/microscopy_png",
    "bids_config": "ivadomed/config/config_bids.json",
    "target_suffix": [["_seg-myelin-manual", "_seg-axon-manual"]],
    "extensions": [".png"],
    "roi_params": {
        "suffix": None,
        "slice_filter_roi": None
    },
    "contrast_params": {
        "training_validation": [],
        "testing": [],
        "balance": {}
    }}])
def test_bids_df_microscopy_png(loader_parameters):
    # Test for microscopy png file format
    # Test for _sessions.tsv and _scans.tsv files
    # Test for target_suffix as a nested list
    # Test for when no contrast_params are provided
    loader_params = loader_parameters
    loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["training_validation"]
    bids_path = loader_params["bids_path"]
    derivatives = True
    df_test = imed_loader_utils.create_bids_dataframe(loader_params, derivatives)
    df_test = df_test.drop(columns=['path', 'parent_path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = os.path.join(bids_path, "df_ref.csv")
    csv_test = os.path.join(bids_path, "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [], 'columns_added': [], 'columns_removed': []}


@pytest.mark.parametrize('loader_parameters', [{
    "bids_path": "testing_data",
    "target_suffix": ["_seg-manual"],
    "extensions": [],
    "roi_params": {
        "suffix": None,
        "slice_filter_roi": None
    },
    "contrast_params": {
        "training_validation": ["T1w", "T2w"],
        "testing": [],
        "balance": {}
    }}])
def test_bids_df_anat(loader_parameters):
    # Test for MRI anat nii.gz file format
    # Test for when no file extensions are provided
    # Test for multiple target_suffix
    # TODO: modify test and "df_ref.csv" file in data-testing dataset to test behavior when "roi_suffix" is not None
    loader_params = loader_parameters
    loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["training_validation"]
    bids_path = loader_params["bids_path"]
    derivatives = True
    df_test = imed_loader_utils.create_bids_dataframe(loader_params, derivatives)
    df_test = df_test.drop(columns=['path', 'parent_path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = os.path.join(bids_path, "df_ref.csv")
    csv_test = os.path.join(bids_path, "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [], 'columns_added': [], 'columns_removed': []}

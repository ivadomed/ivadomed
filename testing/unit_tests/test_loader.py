import os
import subprocess

import pytest
import pandas as pd

from ivadomed.loader import utils as imed_loader_utils


@pytest.mark.parametrize('loader_parameters', [{
    "bids_path": "testing_data/microscopy_png",
    "bids_config": "ivadomed/config/config_bids.json",
    "target_suffix": [["_seg-myelin-manual", "_seg-axon-manual"]],
    "extensions": [".png"],
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
    derivatives = True
    df = imed_loader_utils.create_bids_dataframe(loader_params, derivatives)
    df = df.drop(columns=['path', 'parent_path'])
    df = df.sort_values(by=['filename']).reset_index(drop=True)
    df.to_csv("testing_data/microscopy_png/df_test.csv", index=False)
    command = "diff testing_data/microscopy_png/df_test.csv testing_data/microscopy_png/df_ref.csv"
    subprocess.check_output(command, shell=True)


@pytest.mark.parametrize('loader_parameters', [{
    "bids_path": "testing_data",
    "target_suffix": ["_seg-manual"],
    "extensions": [],
    "contrast_params": {
        "training_validation": ["T1w", "T2w"],
        "testing": [],
        "balance": {}
    }}])
def test_bids_df_anat(loader_parameters):
    # Test for MRI anat nii.gz file format
    # Test for when no file extensions are provided
    # Test for multiple target_suffix
    loader_params = loader_parameters
    loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["training_validation"]
    derivatives = True
    df = imed_loader_utils.create_bids_dataframe(loader_params, derivatives)
    df = df.drop(columns=['path', 'parent_path'])
    df = df.sort_values(by=['filename']).reset_index(drop=True)
    df.to_csv("testing_data/df_test.csv", index=False)
    command = "diff testing_data/df_test.csv testing_data/df_ref.csv"
    subprocess.check_output(command, shell=True)

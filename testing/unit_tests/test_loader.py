import pytest
import csv_diff
import numpy as np

from ivadomed.loader.bids_dataframe import BidsDataframe
from testing.unit_tests.t_utils import create_tmp_dir, __tmp_dir__, download_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.loader import loader as imed_loader
from ivadomed.keywords import MetadataKW
from pytest_cases import parametrize_with_cases
from testing.unit_tests.cases_loader import *


def setup_function():
    create_tmp_dir()


@parametrize_with_cases("loader_parameters", cases=[
    case_bids_df_microscopy_png,
])
def test_bids_df_microscopy_png(download_data_testing_test_files, loader_parameters):
    """
    Test for microscopy png file format
    Test for _sessions.tsv and _scans.tsv files
    Test for target_suffix as a nested list
    Test for when no contrast_params are provided
    """

    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = Path(loader_parameters["path_data"][0], "df_ref.csv")
    csv_test = Path(loader_parameters["path_data"][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [], 'columns_added': [], 'columns_removed': []}


@parametrize_with_cases("loader_parameters", cases=[
    case_bids_df_anat,
])
def test_bids_df_anat(download_data_testing_test_files, loader_parameters):
    """
    Test for MRI anat nii.gz file format
    Test for when no file extensions are provided
    Test for multiple target_suffix
    Test behavior when "roi_suffix" is not None
    """

    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = Path(loader_parameters["path_data"][0], "df_ref.csv")
    csv_test = Path(loader_parameters["path_data"][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [],
                    'columns_added': [], 'columns_removed': []}


@parametrize_with_cases("loader_parameters", cases=[
    case_bids_df_multi
])
def test_bids_df_multi(download_data_testing_test_files, loader_parameters):
    """
    Test for multiple folders in path_data
    """

    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = Path(loader_parameters["path_data"][0], "df_ref_multi.csv")
    csv_test = Path(loader_parameters["path_data"][0], "df_test_multi.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [],
                    'columns_added': [], 'columns_removed': []}


@parametrize_with_cases("loader_parameters", cases=[
    case_bids_df_ctscan,
])
def test_bids_df_ctscan(download_data_testing_test_files, loader_parameters):
    """
    Test for ct-scan nii.gz file format
    Test for when dataset_description.json is not present in derivatives folder
    """

    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = Path(loader_parameters["path_data"][0], "df_ref.csv")
    csv_test = Path(loader_parameters["path_data"][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [], 'columns_added': [], 'columns_removed': []}


@parametrize_with_cases("seg_pair", cases=[
    case_dropout_input_2_5_5,
    case_dropout_input_1_5_5,
    case_dropout_input_5_5_5_5,
    case_dropout_input_5_5_5_3,
    case_dropout_input_7_7_4
])
def test_dropout_input(seg_pair):
    n_channels = seg_pair['input'].size(0)
    seg_pair = imed_loader_utils.dropout_input(seg_pair)
    empty_channels = [len(torch.unique(input_data)) == 1 for input_data in seg_pair['input']]

    # If multichannel
    if n_channels > 1:
        # Verify that there is still at least one channel remaining
        assert sum(empty_channels) <= n_channels
    else:
        assert sum(empty_channels) == 0


@parametrize_with_cases("loader_parameters, model_parameters, transform_parameters", cases=[
    case_load_dataset_2d_png,
])
def test_load_dataset_2d_png(download_data_testing_test_files,
                             loader_parameters, model_parameters, transform_parameters):
    """
    Test to make sure load_dataset runs with 2D PNG files, writes corresponding NIfTI files,
    and binarizes ground-truth values to 0 and 1.
    """
    loader_parameters.update({"model_params": model_parameters})
    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    data_lst = ['sub-rat3_ses-01_sample-data9_SEM.png']
    ds = imed_loader.load_dataset(bids_df,
                                  **{**loader_parameters, **{'data_list': data_lst,
                                                             'transforms_params': transform_parameters,
                                                             'dataset_type': 'training'}})
    fname_png = bids_df.df[bids_df.df['filename'] == data_lst[0]]['path'].values[0]
    fname_nii = imed_loader_utils.update_filename_to_nifti(fname_png)
    assert Path(fname_nii).exists() == 1
    assert ds[0]['input'].shape == (1, 756, 764)
    assert ds[0]['gt'].shape == (1, 756, 764)
    assert np.unique(ds[0]['gt']).tolist() == [0, 1]


@parametrize_with_cases("loader_parameters, model_parameters, transform_parameters", cases=[
    case_2d_patches_and_resampling,
])
def test_2d_patches_and_resampling(download_data_testing_test_files,
                                   loader_parameters, model_parameters, transform_parameters):
    """
    Test that 2d patching is done properly.
    Test that microscopy pixelsize and resampling are applied on the right dimensions.
    """
    loader_parameters.update({"model_params": model_parameters})
    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    data_lst = ['sub-rat3_ses-01_sample-data9_SEM.png']
    ds = imed_loader.load_dataset(bids_df,
                                  **{**loader_parameters, **{'data_list': data_lst,
                                                             'transforms_params': transform_parameters,
                                                             'dataset_type': 'training'}})
    assert ds.is_2d_patch == True
    assert ds[0]['input'].shape == (1, 256, 128)
    assert ds[0]['input_metadata'][0].metadata[MetadataKW.INDEX_SHAPE] == (1512, 382)
    assert len(ds) == 28


@parametrize_with_cases("loader_parameters, model_parameters, transform_parameters", cases=[
    case_get_target_filename_list,
])
def test_get_target_filename_list(loader_parameters, model_parameters, transform_parameters):
    """
    Test that all target_suffix are considered for target filename when list
    """
    loader_parameters.update({"model_params": model_parameters})
    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    data_lst = ['sub-rat3_ses-01_sample-data9_SEM.png']
    test_ds = imed_loader.load_dataset(bids_df,
                                       **{**loader_parameters, **{'data_list': data_lst,
                                                                  'transforms_params': transform_parameters,
                                                                  'dataset_type': 'training'}})
    target_filename = test_ds.filename_pairs[0][1]
    
    assert len(target_filename) == len(loader_parameters["target_suffix"])


@parametrize_with_cases("loader_parameters, model_parameters, transform_parameters", cases=[
    case_get_target_filename_list_multiple_raters,
])
def test_get_target_filename_list_multiple_raters(loader_parameters, model_parameters, transform_parameters):
    """
    Test that all target_suffix are considered for target filename when list
    """
    loader_parameters.update({"model_params": model_parameters})
    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    data_lst = ['sub-rat3_ses-01_sample-data9_SEM.png']
    test_ds = imed_loader.load_dataset(bids_df,
                                       **{**loader_parameters, **{'data_list': data_lst,
                                                                  'transforms_params': transform_parameters,
                                                                  'dataset_type': 'training'}})
    target_filename = test_ds.filename_pairs[0][1]

    assert len(target_filename) == len(loader_parameters["target_suffix"])
    assert len(target_filename[0]) == len(loader_parameters["target_suffix"][0])
    assert len(target_filename[1]) == len(loader_parameters["target_suffix"][1])


def teardown_function():
    remove_tmp_dir()

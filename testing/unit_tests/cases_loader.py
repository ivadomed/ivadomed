from pathlib import Path

import torch

from testing.unit_tests.t_utils import __data_testing_dir__, path_repo_root
from ivadomed.keywords import LoaderParamsKW, ModelParamsKW, TransformationKW


def get_microscopy_loader_parameters():
    default_loader_parameters = {
        LoaderParamsKW.PATH_DATA: [str(Path(__data_testing_dir__, "microscopy_png"))],
        LoaderParamsKW.BIDS_CONFIG: f"{path_repo_root}/ivadomed/config/config_bids.json",
        LoaderParamsKW.TARGET_SUFFIX: [["_seg-myelin-manual", "_seg-axon-manual"]],
        LoaderParamsKW.EXTENSIONS: [".png"],
        LoaderParamsKW.ROI_PARAMS: {"suffix": None, "slice_filter_roi": None},
        LoaderParamsKW.CONTRAST_PARAMS: {"contrast_lst": []}
    }

    return default_loader_parameters


def case_bids_df_microscopy_png():
    loader_parameters = get_microscopy_loader_parameters()

    return loader_parameters


def case_bids_df_anat():
    loader_parameters = {
        LoaderParamsKW.PATH_DATA: [__data_testing_dir__],
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual"],
        LoaderParamsKW.EXTENSIONS: [],
        LoaderParamsKW.ROI_PARAMS: {"suffix": "_seg-manual", "slice_filter_roi": None},
        LoaderParamsKW.CONTRAST_PARAMS: {"contrast_lst": ["T1w", "T2w"]}
    }

    return loader_parameters


def case_bids_df_multi():
    loader_parameters = get_microscopy_loader_parameters()
    loader_parameters.update({
        LoaderParamsKW.PATH_DATA: [__data_testing_dir__, str(Path(__data_testing_dir__, "microscopy_png"))],
        LoaderParamsKW.TARGET_SUFFIX: ["_seg-manual", "seg-axon-manual"],
        LoaderParamsKW.EXTENSIONS: [".nii.gz", ".png"],
        LoaderParamsKW.CONTRAST_PARAMS: {"contrast_lst": ["T1w", "T2w", "SEM"]}
    })

    return loader_parameters


def case_bids_df_ctscan():
    loader_parameters = {
        LoaderParamsKW.PATH_DATA: [str(Path(__data_testing_dir__, "ct_scan"))],
        LoaderParamsKW.BIDS_CONFIG: f"{path_repo_root}/ivadomed/config/config_bids.json",
        LoaderParamsKW.TARGET_SUFFIX: ["_seg-manual"],
        LoaderParamsKW.EXTENSIONS: [".nii.gz"],
        LoaderParamsKW.ROI_PARAMS: {"suffix": None, "slice_filter_roi": None},
        LoaderParamsKW.CONTRAST_PARAMS: {"contrast_lst": ["ct"]}
    }

    return loader_parameters


def case_dropout_input_2_5_5():
    seg_pair = {"input": torch.rand((2, 5, 5))}
    return seg_pair


def case_dropout_input_1_5_5():
    seg_pair = {"input": torch.rand((1, 5, 5))}
    return seg_pair


def case_dropout_input_5_5_5_5():
    seg_pair = {"input": torch.rand((5, 5, 5, 5))}
    return seg_pair


def case_dropout_input_5_5_5_3():
    seg_pair = {"input": (torch.rand((5, 5, 5, 3)) * torch.tensor([1, 0, 1], dtype=torch.float)).transpose(0, -1)}
    return seg_pair


def case_dropout_input_7_7_4():
    seg_pair = {"input": (torch.rand((7, 7, 4)) * torch.tensor([1, 0, 0, 0], dtype=torch.float)).transpose(0, -1)}
    return seg_pair


def case_load_dataset_2d_png():
    loader_parameters = get_microscopy_loader_parameters()
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_seg-myelin-manual"],
        LoaderParamsKW.CONTRAST_PARAMS: {"contrast_lst": [], "balance": {}},
        LoaderParamsKW.SLICE_AXIS: "axial",
        LoaderParamsKW.SLICE_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": True},
        LoaderParamsKW.PATCH_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": False},
        LoaderParamsKW.MULTICHANNEL: False
    })

    model_parameters = {
        ModelParamsKW.NAME: "Unet",
        ModelParamsKW.DROPOUT_RATE: 0.3,
        ModelParamsKW.BN_MOMENTUM: 0.1,
        ModelParamsKW.FINAL_ACTIVATION: "sigmoid",
        ModelParamsKW.DEPTH: 3
    }

    transform_parameters = {
        TransformationKW.NUMPY_TO_TENSOR: {},
    }

    return loader_parameters, model_parameters, transform_parameters


def case_2d_patches_and_resampling():
    loader_parameters = get_microscopy_loader_parameters()
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_seg-myelin-manual"],
        LoaderParamsKW.CONTRAST_PARAMS: {"contrast_lst": [], "balance": {}},
        LoaderParamsKW.SLICE_AXIS: "axial",
        LoaderParamsKW.SLICE_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": True},
        LoaderParamsKW.PATCH_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": False},
        LoaderParamsKW.MULTICHANNEL: False
    })

    model_parameters = {
        ModelParamsKW.NAME: "Unet",
        ModelParamsKW.DROPOUT_RATE: 0.3,
        ModelParamsKW.BN_MOMENTUM: 0.1,
        ModelParamsKW.FINAL_ACTIVATION: "sigmoid",
        ModelParamsKW.DEPTH: 3,
        ModelParamsKW.LENGTH_2D: [256, 128],
        ModelParamsKW.STRIDE_2D: [244, 116]
    }

    transform_parameters = {
        TransformationKW.RESAMPLE: {
            TransformationKW.W_SPACE: 0.0002,
            TransformationKW.H_SPACE: 0.0001
        },
        TransformationKW.NUMPY_TO_TENSOR: {},
    }

    return loader_parameters, model_parameters, transform_parameters


def case_get_target_filename_list():
    loader_parameters = get_microscopy_loader_parameters()
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_seg-myelin-manual", "_seg-axon-manual"],
        LoaderParamsKW.CONTRAST_PARAMS: {"contrast_lst": [], "balance": {}},
        LoaderParamsKW.SLICE_AXIS: "axial",
        LoaderParamsKW.SLICE_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": True},
        LoaderParamsKW.PATCH_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": False},
        LoaderParamsKW.MULTICHANNEL: False
    })

    model_parameters = {
        ModelParamsKW.NAME: "Unet",
        ModelParamsKW.DROPOUT_RATE: 0.3,
        ModelParamsKW.BN_MOMENTUM: 0.1,
        ModelParamsKW.DEPTH: 2
    }

    transform_parameters = {
        TransformationKW.NUMPY_TO_TENSOR: {},
    }

    return loader_parameters, model_parameters, transform_parameters


def case_get_target_filename_list_multiple_raters():
    loader_parameters = get_microscopy_loader_parameters()
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: [["_seg-myelin-manual", "_seg-axon-manual"],
                                       ["_seg-myelin-manual", "_seg-axon-manual"]],
        LoaderParamsKW.CONTRAST_PARAMS: {"contrast_lst": [], "balance": {}},
        LoaderParamsKW.SLICE_AXIS: "axial",
        LoaderParamsKW.SLICE_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": True},
        LoaderParamsKW.PATCH_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": False},
        LoaderParamsKW.MULTICHANNEL: False
    })

    model_parameters = {
        ModelParamsKW.NAME: "Unet",
        ModelParamsKW.DROPOUT_RATE: 0.3,
        ModelParamsKW.BN_MOMENTUM: 0.1,
        ModelParamsKW.DEPTH: 2
    }

    transform_parameters = {
        TransformationKW.NUMPY_TO_TENSOR: {},
    }

    return loader_parameters, model_parameters, transform_parameters


def case_microscopy_pixelsize():
    loader_parameters = get_microscopy_loader_parameters()
    loader_parameters.update({
        LoaderParamsKW.TARGET_SUFFIX: ["_seg-myelin-manual"],
        LoaderParamsKW.CONTRAST_PARAMS: {"contrast_lst": [], "balance": {}},
        LoaderParamsKW.SLICE_AXIS: "axial",
        LoaderParamsKW.SLICE_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": True},
        LoaderParamsKW.PATCH_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": False},
        LoaderParamsKW.MULTICHANNEL: False
    })

    model_parameters = {
        ModelParamsKW.NAME: "Unet",
        ModelParamsKW.DROPOUT_RATE: 0.3,
        ModelParamsKW.BN_MOMENTUM: 0.1,
        ModelParamsKW.FINAL_ACTIVATION: "sigmoid",
        ModelParamsKW.DEPTH: 3
    }

    return loader_parameters, model_parameters


def case_read_png_tif():
    loader_parameters = {
        LoaderParamsKW.PATH_DATA: [str(Path(__data_testing_dir__, "data_test_png_tif"))],
        LoaderParamsKW.BIDS_CONFIG: f"{path_repo_root}/ivadomed/config/config_bids.json",
        LoaderParamsKW.TARGET_SUFFIX: ["_seg-myelin-manual"],
        LoaderParamsKW.EXTENSIONS: [".png", ".tif"],
        LoaderParamsKW.ROI_PARAMS: {"suffix": None, "slice_filter_roi": None},
        LoaderParamsKW.CONTRAST_PARAMS: {"contrast_lst": [], "balance": {}},
        LoaderParamsKW.SLICE_AXIS: "axial",
        LoaderParamsKW.SLICE_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": True},
        LoaderParamsKW.PATCH_FILTER_PARAMS: {"filter_empty_mask": False, "filter_empty_input": False},
        LoaderParamsKW.MULTICHANNEL: False
    }
    model_parameters = {
        ModelParamsKW.NAME: "Unet",
        ModelParamsKW.DROPOUT_RATE: 0.3,
        ModelParamsKW.BN_MOMENTUM: 0.1,
        ModelParamsKW.FINAL_ACTIVATION: "sigmoid",
        ModelParamsKW.DEPTH: 3
    }

    return loader_parameters, model_parameters

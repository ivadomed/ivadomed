import numpy as np
import pytest
import os
import json
import shutil

from ivadomed.loader import loader as imed_loader
from ivadomed.object_detection import utils as imed_obj_detect

PATH_BIDS = 'testing_data'
BATCH_SIZE = 8
LOG_DIR = "log"


@pytest.mark.parametrize('train_lst', [['sub-unf01']])
@pytest.mark.parametrize('target_lst', [["_lesion-manual"]])
@pytest.mark.parametrize('config', [
    {
        "object_detection_params": {
            "object_detection_path": "object_detection",
            "safety_factor": [1.0, 1.0, 1.0],
            "log_directory": LOG_DIR
        },
        "transforms_params": {
            "NumpyToTensor": {}},
        "roi_params": {"suffix": "_seg-manual", "slice_filter_roi": 10},
        "contrast_params": {"contrast_lst": ['T2w'], "balance": {}},
        "multichannel": False,
        "model_params": {"name": "Unet"},
    }, {
        "object_detection_params": {
            "object_detection_path": "object_detection",
            "safety_factor": [1.0, 1.0, 1.0],
            "log_directory": LOG_DIR
        },
        "transforms_params": {"NumpyToTensor": {}},
        "roi_params": {"suffix": "_seg-manual", "slice_filter_roi": 10},
        "contrast_params": {"contrast_lst": ['T2w'], "balance": {}},
        "Modified3DUNet": {
            "applied": True,
            "length_3D": [16, 16, 16],
            "stride_3D": [1, 1, 1],
            "attention": False,
            "n_filters": 8
        },
        "multichannel": False,
        "model_params": {"name": "Unet"},
    }])
def test_bounding_box(train_lst, target_lst, config):
    # Create mask
    mask_coord = [20, 40, 20, 90, 0, 25]
    mx1, mx2, my1, my2, mz1, mz2 = mask_coord
    mask = np.zeros((96, 96, 96))
    mask[mx1:mx2 + 1, my1:my2 + 1, mz1:mz2 + 1] = 1
    coord = imed_obj_detect.get_bounding_boxes(mask)
    assert coord[0] == mask_coord

    loader_params = {
        "data_list": train_lst,
        "dataset_type": "training",
        "requires_undo": False,
        "bids_path": PATH_BIDS,
        "target_suffix": target_lst,
        "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
        "slice_axis": "axial"
    }

    if "Modified3DUNet" in config:
        config['model_params']["name"] = "Modified3DUNet"
        config['model_params'].update(config["Modified3DUNet"])

    bounding_box_dict = {}
    bounding_box_path = os.path.join(LOG_DIR, 'bounding_boxes.json')
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    current_dir = os.getcwd()
    sub = train_lst[0]
    contrast = config['contrast_params']['contrast_lst'][0]
    bb_path = os.path.join(current_dir, PATH_BIDS, sub, "anat", sub + "_" + contrast + ".nii.gz")
    bounding_box_dict[bb_path] = coord
    with open(bounding_box_path, 'w') as fp:
        json.dump(bounding_box_dict, fp, indent=4)

    # Update loader_params with config
    loader_params.update(config)
    ds = imed_loader.load_dataset(**loader_params)

    handler = ds.handlers if "Modified3DUNet" in config else ds.indexes
    for index in handler:
        seg_pair, _ = index
        if "Modified3DUNet" in config:
            assert seg_pair['input'][0].shape[-3:] == (mx2 - mx1, my2 - my1, mz2 - mz1)
        else:
            assert seg_pair['input'][0].shape[-2:] == (mx2 - mx1, my2 - my1)

    shutil.rmtree(LOG_DIR)


# testing adjust bb size
def test_adjust_bb_size():
    test_coord = (0, 10, 0, 10, 0, 10)
    res = imed_obj_detect.adjust_bb_size(test_coord, (2, 2, 2), True)
    assert(res == [0, 20, 0, 20, 0, 20])


# testing bb statistic.
# We just check wether or it is running as it not returning anything
def test_compute_bb_statistics():
    imed_obj_detect.compute_bb_statistics("testing_data/bounding_box_dict.json")


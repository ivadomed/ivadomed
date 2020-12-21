import pytest
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader

cudnn.benchmark = True

GPU_NUMBER = 0
BATCH_SIZE = 1
PATH_BIDS = 'testing_data'


def _cmpt_slice(ds_loader):
    cmpt_label = {0: 0, 1: 0}
    for i, batch in enumerate(ds_loader):
        for gt in batch['gt']:
            # TODO: multi label
            if np.any(gt.numpy()):
                cmpt_label[1] += 1
            else:
                cmpt_label[0] += 1
    print(cmpt_label)
    return cmpt_label[0], cmpt_label[1]


@pytest.mark.parametrize('transforms_dict', [
    {"Resample": {"wspace": 0.75, "hspace": 0.75},
     "ROICrop": {"size": [48, 48]},
     "NumpyToTensor": {}},
    {"Resample": {"wspace": 0.75, "hspace": 0.75, "applied_to": ["im", "gt"]},
     "CenterCrop": {"size": [100, 100], "applied_to": ["im", "gt"]},
     "NumpyToTensor": {"applied_to": ["im", "gt"]}}])
@pytest.mark.parametrize('train_lst', [['sub-unf01']])
@pytest.mark.parametrize('target_lst', [["_lesion-manual"]])
@pytest.mark.parametrize('slice_filter_params', [
    {"filter_empty_mask": False, "filter_empty_input": True},
    {"filter_empty_mask": True, "filter_empty_input": True}])
@pytest.mark.parametrize('roi_params', [
    {"suffix": "_seg-manual", "slice_filter_roi": 10},
    {"suffix": None, "slice_filter_roi": 0}])
def test_slice_filter(transforms_dict, train_lst, target_lst, roi_params, slice_filter_params):
    if "ROICrop" in transforms_dict and roi_params["suffix"] == None:
        return

    cuda_available, device = imed_utils.define_device(GPU_NUMBER)

    loader_params = {
        "transforms_params": transforms_dict,
        "data_list": train_lst,
        "dataset_type": "training",
        "requires_undo": False,
        "contrast_params": {"contrast_lst": ['T2w'], "balance": {}},
        "bids_path": PATH_BIDS,
        "target_suffix": target_lst,
        "roi_params": roi_params,
        "model_params": {"name": "Unet"},
        "slice_filter_params": slice_filter_params,
        "slice_axis": "axial",
        "multichannel": False
    }
    # Get Training dataset
    ds_train = imed_loader.load_dataset(**loader_params)

    print('\tNumber of loaded slices: {}'.format(len(ds_train)))

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=0)
    print('\tNumber of Neg/Pos slices in GT.')
    cmpt_neg, cmpt_pos = _cmpt_slice(train_loader)
    if slice_filter_params["filter_empty_mask"]:
        assert cmpt_neg == 0
        assert cmpt_pos != 0
    else:
        # We verify if there are still some negative slices (they are removed with our filter)
        assert cmpt_neg != 0 and cmpt_pos != 0

import numpy as np
import pytest
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader

cudnn.benchmark = True

GPU_NUMBER = 0
BATCH_SIZE = 8
PATH_BIDS = 'testing_data'


def _cmpt_label(ds_loader):
    cmpt_label, cmpt_sample = {0: 0, 1: 0}, 0
    for i, batch in enumerate(ds_loader):
        for gt in batch['gt']:
            for idx in range(len(gt)):
                if np.any(gt[idx].numpy()[0]):
                    cmpt_label[1] += 1
                else:
                    cmpt_label[0] += 1
                cmpt_sample += 1

    neg_sample_ratio = cmpt_label[0] * 100. / cmpt_sample
    pos_sample_ratio = cmpt_label[1] * 100. / cmpt_sample
    print({'neg_sample_ratio': neg_sample_ratio,
           'pos_sample_ratio': pos_sample_ratio})


@pytest.mark.parametrize('transforms_dict', [{
    "Resample":
        {
            "wspace": 0.75,
            "hspace": 0.75
        },
    "ROICrop":
        {
            "size": [48, 48]
        },
    "NumpyToTensor": {}
    }])
@pytest.mark.parametrize('train_lst', [['sub-test001']])
@pytest.mark.parametrize('target_lst', [["_seg-manual"]])
@pytest.mark.parametrize('roi_params', [{"suffix": "_seg-manual", "slice_filter_roi": 10}])
def test_sampler(transforms_dict, train_lst, target_lst, roi_params):
    cuda_available, device = imed_utils.define_device(GPU_NUMBER)

    loader_params = {
        "transforms_params": transforms_dict,
        "data_list": train_lst,
        "dataset_type": "training",
        "requires_undo": False,
        "contrast_lst": ['T2w'],
        "balance": {},
        "bids_path": PATH_BIDS,
        "target_suffix": target_lst,
        "roi_params": roi_params,
        "slice_filter_params": {
            "filter_empty_mask": False,
            "filter_empty_input": True
        },
        "slice_axis": "axial",
        "multichannel": False
    }
    # Get Training dataset
    ds_train = imed_loader.load_dataset(**loader_params)

    print('\nLoading without sampling')
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=0)
    _cmpt_label(train_loader)

    print('\nLoading with sampling')
    train_loader_balanced = DataLoader(ds_train, batch_size=BATCH_SIZE,
                                       sampler=imed_loader_utils.BalancedSampler(ds_train),
                                       shuffle=False, pin_memory=True,
                                       collate_fn=imed_loader_utils.imed_collate,
                                       num_workers=0)
    _cmpt_label(train_loader_balanced)

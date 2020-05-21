import itertools

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import ivadomed.transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader

cudnn.benchmark = True

GPU_NUMBER = 0
BATCH_SIZE = 8
PATH_BIDS = 'testing_data'


def _cmpt_slice(ds_loader, gt_roi='gt'):
    cmpt_label, cmpt_sample = {0: 0, 1: 0}, 0
    for i, batch in enumerate(ds_loader):
        for idx in range(len(batch[gt_roi])):
            smp_np = batch[gt_roi][idx].numpy()
            # For now only supports 1 label
            if np.any(smp_np[0, :]):
                cmpt_label[1] += 1
            else:
                cmpt_label[0] += 1
            cmpt_sample += 1
    print(cmpt_label)


def test_slice_filter_center():
    """Test SliceFilter when using mt_transforms.CenterCrop2D."""
    device = torch.device("cuda:" + str(GPU_NUMBER) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(device)
        print("Using GPU number {}".format(device))

    training_transform_dict = {
        "Resample":
            {
                "wspace": 0.75,
                "hspace": 0.75
            },
        "CenterCrop":
            {
                "size": [100, 100]
            }
    }

    train_transform = imed_transforms.Compose(training_transform_dict)

    train_lst = ['sub-test001']
    roi_lst = [None]
    empty_input_lst = [True]
    empty_mask_lst = [True, False]
    param_lst_lst = [roi_lst, empty_input_lst, empty_mask_lst]
    for roi, empty_input, empty_mask in list(itertools.product(*param_lst_lst)):
        print('\nROI: {}, Empty Input: {}, Empty Mask: {}'.format(roi, empty_input, empty_mask))
        ds_train = imed_loader.BidsDataset(PATH_BIDS,
                                           subject_lst=train_lst,
                                           target_suffix=["_lesion-manual"],
                                           roi_suffix=roi,
                                           contrast_lst=['T2w'],
                                           metadata_choice="without",
                                           contrast_balance={},
                                           slice_axis=2,
                                           transform=train_transform,
                                           multichannel=False,
                                           slice_filter_fn=imed_utils.SliceFilter(filter_empty_input=empty_input,
                                                                                  filter_empty_mask=empty_mask))

        print('\tNumber of loaded slices: {}'.format(len(ds_train)))
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory=True,
                                  collate_fn=imed_loader_utils.imed_collate,
                                  num_workers=0)
        print('\tNumber of Neg/Pos slices in GT.')
        _cmpt_slice(train_loader, 'gt')


def test_slice_filter_roi():
    """Test SliceFilter when using ivadomed_transforms.ROICrop2D."""
    device = torch.device("cuda:" + str(GPU_NUMBER) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(device)
        print("Using GPU number {}".format(device))

    training_transform_dict = {
        "Resample":
            {
                "wspace": 0.75,
                "hspace": 0.75
            },
        "ROICrop":
            {
                "size": [100, 100]
            }
    }

    train_transform = imed_transforms.Compose(training_transform_dict)

    train_lst = ['sub-test001']
    roi_lst = ['_seg-manual']
    empty_input_lst = [True]
    empty_mask_lst = [True, False]
    param_lst_lst = [roi_lst, empty_input_lst, empty_mask_lst]
    for roi, empty_input, empty_mask in list(itertools.product(*param_lst_lst)):
        print('\nROI: {}, Empty Input: {}, Empty Mask: {}'.format(roi, empty_input, empty_mask))
        ds_train = imed_loader.BidsDataset(PATH_BIDS,
                                           subject_lst=train_lst,
                                           target_suffix=["_lesion-manual"],
                                           roi_suffix=roi,
                                           contrast_lst=['T2w'],
                                           metadata_choice="without",
                                           contrast_balance={},
                                           slice_axis=2,
                                           transform=train_transform,
                                           multichannel=False,
                                           slice_filter_fn=imed_utils.SliceFilter(filter_empty_input=empty_input,
                                                                                  filter_empty_mask=empty_mask))

        print('\tNumber of loaded slices before filtering ROI: {}'.format(len(ds_train)))
        ds_train = imed_loader_utils.filter_roi(ds_train, nb_nonzero_thr=10)

        print('\tNumber of loaded slices: {}'.format(len(ds_train)))
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory=True,
                                  collate_fn=imed_loader_utils.imed_collate,
                                  num_workers=0)
        print('\tNumber of Neg/Pos slices in GT.')
        _cmpt_slice(train_loader, 'gt')
        print('\tNumber of Neg/Pos slices in ROI.')
        _cmpt_slice(train_loader, 'roi')

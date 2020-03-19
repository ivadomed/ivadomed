import numpy as np
import itertools

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from ivadomed.utils import SliceFilter
from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms

from ivadomed import loader as loader
import ivadomed.transforms as ivadomed_transforms

cudnn.benchmark = True

GPU_NUMBER = 0
BATCH_SIZE = 8
PATH_BIDS = 'testing_data'


def _cmpt_slice(ds_loader, gt_roi='gt'):
    cmpt_label, cmpt_sample = {0: 0, 1: 0}, 0
    for i, batch in enumerate(ds_loader):
        gt_samples = batch[gt_roi]
        for idx in range(len(gt_samples)):
            if np.any(gt_samples[idx]):
                cmpt_label[1] += 1
            else:
                cmpt_label[0] += 1
            cmpt_sample += 1
    print(cmpt_label)


def test_slice_filter_center():
    """Test SliceFilter when using mt_transforms.CenterCrop2D."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(GPU_NUMBER)
        print("Using GPU number {}".format(GPU_NUMBER))

    training_transform_list = [
        ivadomed_transforms.Resample(wspace=0.75, hspace=0.75),
        mt_transforms.CenterCrop2D(size=[100, 100])
    ]
    train_transform = transforms.Compose(training_transform_list)

    train_lst = ['sub-test001']
    roi_lst = [None]
    empty_input_lst = [True]
    empty_mask_lst = [True, False]
    param_lst_lst = [roi_lst, empty_input_lst, empty_mask_lst]
    for roi, empty_input, empty_mask in list(itertools.product(*param_lst_lst)):
        print('\nROI: {}, Empty Input: {}, Empty Mask: {}'.format(roi, empty_input, empty_mask))
        ds_train = loader.BidsDataset(PATH_BIDS,
                                      subject_lst=train_lst,
                                      target_suffix="_lesion-manual",
                                      roi_suffix=roi,
                                      contrast_lst=['T2w'],
                                      metadata_choice="without",
                                      contrast_balance={},
                                      slice_axis=2,
                                      transform=train_transform,
                                      multichannel=False,
                                      slice_filter_fn=SliceFilter(filter_empty_input=empty_input,
                                                                  filter_empty_mask=empty_mask))

        print('\tNumber of loaded slices: {}'.format(len(ds_train)))
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory=True,
                                  collate_fn=mt_datasets.mt_collate,
                                  num_workers=0)
        print('\tNumber of Neg/Pos slices in GT.')
        _cmpt_slice(train_loader, 'gt')


def test_slice_filter_roi():
    """Test SliceFilter when using ivadomed_transforms.ROICrop2D."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(GPU_NUMBER)
        print("Using GPU number {}".format(GPU_NUMBER))

    training_transform_list = [
        ivadomed_transforms.Resample(wspace=0.75, hspace=0.75),
        ivadomed_transforms.ROICrop2D(size=[100, 100])
    ]
    train_transform = transforms.Compose(training_transform_list)

    train_lst = ['sub-test001']
    roi_lst = ['_seg-manual']
    empty_input_lst = [True]
    empty_mask_lst = [True, False]
    param_lst_lst = [roi_lst, empty_input_lst, empty_mask_lst]
    for roi, empty_input, empty_mask in list(itertools.product(*param_lst_lst)):
        print('\nROI: {}, Empty Input: {}, Empty Mask: {}'.format(roi, empty_input, empty_mask))
        ds_train = loader.BidsDataset(PATH_BIDS,
                                      subject_lst=train_lst,
                                      target_suffix="_lesion-manual",
                                      roi_suffix=roi,
                                      contrast_lst=['T2w'],
                                      metadata_choice="without",
                                      contrast_balance={},
                                      slice_axis=2,
                                      transform=train_transform,
                                      multichannel=False,
                                      slice_filter_fn=SliceFilter(filter_empty_input=empty_input,
                                                                  filter_empty_mask=empty_mask))

        print('\tNumber of loaded slices before filtering ROI: {}'.format(len(ds_train)))
        ds_train = loader.filter_roi(ds_train, nb_nonzero_thr=10)

        print('\tNumber of loaded slices: {}'.format(len(ds_train)))
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory=True,
                                  collate_fn=mt_datasets.mt_collate,
                                  num_workers=0)
        print('\tNumber of Neg/Pos slices in GT.')
        _cmpt_slice(train_loader, 'gt')
        print('\tNumber of Neg/Pos slices in ROI.')
        _cmpt_slice(train_loader, 'roi')


print("Test test_slice_filter_center")
test_slice_filter_center()

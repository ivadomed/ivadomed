import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as torch_transforms

import ivadomed.transforms as imed_transforms
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
                if np.any(gt[idx]):
                    cmpt_label[1] += 1
                else:
                    cmpt_label[0] += 1
                cmpt_sample += 1

    neg_sample_ratio = cmpt_label[0] * 100. / cmpt_sample
    pos_sample_ratio = cmpt_label[1] * 100. / cmpt_sample
    print({'neg_sample_ratio': neg_sample_ratio,
           'pos_sample_ratio': pos_sample_ratio})


def test_sampler():
    device = torch.device("cuda:"+str(GPU_NUMBER) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(device)
        print("Using GPU number {}".format(device))

    training_transform_list = [
        imed_transforms.Resample(wspace=0.75, hspace=0.75),
        imed_transforms.ROICrop2D(size=[48, 48])
    ]
    train_transform = torch_transforms.Compose(training_transform_list)

    train_lst = ['sub-test001']

    ds_train = imed_loader.BidsDataset(PATH_BIDS,
                                       subject_lst=train_lst,
                                       target_suffix=["_lesion-manual"],
                                       roi_suffix="_seg-manual",
                                       contrast_lst=['T2w'],
                                       metadata_choice="without",
                                       contrast_balance={},
                                       slice_axis=2,
                                       transform=train_transform,
                                       multichannel=False,
                                       slice_filter_fn=imed_utils.SliceFilter(filter_empty_input=True,
                                                                              filter_empty_mask=False))

    ds_train = imed_loader_utils.filter_roi(ds_train, nb_nonzero_thr=10)

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


print("Test sampler")
test_sampler()

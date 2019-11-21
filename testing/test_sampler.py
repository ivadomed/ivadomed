import numpy as np

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms

from ivadomed import loader as loader
from ivadomed.utils import *
import ivadomed.transforms as ivadomed_transforms

cudnn.benchmark = True

GPU_NUMBER = 0
BATCH_SIZE = 8
PATH_BIDS = '../duke/projects/ivado-medical-imaging/testing_data/lesion_data/'

def test_sampler():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(GPU_NUMBER)
        print("Using GPU number {}".format(GPU_NUMBER))

    training_transform_list = [
        ivadomed_transforms.Resample(wspace=0.75, hspace=0.75),
        ivadomed_transforms.ROICrop2D(size=[48, 48])
    ]
    train_transform = transforms.Compose(training_transform_list)

    train_lst = ['sub-bwh025']

    ds_train = loader.BidsDataset(PATH_BIDS,
                                  subject_lst=train_lst,
                                  target_suffix="_lesion-manual",
                                  roi_suffix="_seg-manual",
                                  contrast_lst=['acq-sagstir_T2w'], #['acq-ax_T2w'],
                                  metadata_choice="without",
                                  contrast_balance={},
                                  slice_axis=2,
                                  transform=train_transform,
                                  multichannel=False,
                                  slice_filter_fn=SliceFilter(nb_nonzero_thr=10))

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=0)

    cmpt_label = {0: 0, 1: 0}
    for i, batch in enumerate(train_loader):
        gt_samples = batch["gt"]
        for idx in range(len(gt_samples)):
            if np.any(gt_samples[idx]):
                cmpt_label[1] += 1
            else:
                cmpt_label[0] += 1
    print(cmpt_label)

    train_loader_balanced = DataLoader(ds_train, batch_size=8,
                                          sampler=loader.BalancedSampler(ds_train),
                                          shuffle=False, pin_memory=True,
                                          collate_fn=mt_datasets.mt_collate,
                                          num_workers=0)

    cmpt_label = {0: 0, 1: 0}
    for i, batch in enumerate(train_loader_balanced):
        gt_samples = batch["gt"]
        for idx in range(len(gt_samples)):
            if np.any(gt_samples[idx]):
                cmpt_label[1] += 1
            else:
                cmpt_label[0] += 1
    print(cmpt_label)


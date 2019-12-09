import numpy as np
import time
import sys

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from medicaltorch.filters import SliceFilter
from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms
from torch import optim

from tqdm import tqdm

from ivadomed import loader as loader
from ivadomed import models
from ivadomed import losses
from ivadomed.utils import *
import ivadomed.transforms as ivadomed_transforms

cudnn.benchmark = True

GPU_NUMBER = 5
BATCH_SIZE = 8
DROPOUT = 0.4
BN = 0.1
N_EPOCHS = 10
INIT_LR = 0.01
PATH_BIDS = 'testing_data/'

def test_dilateGT():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(GPU_NUMBER)
        print("Using GPU number {}".format(GPU_NUMBER))

    training_transform_list = [
        ivadomed_transforms.Resample(wspace=0.75, hspace=0.75),
        ivadomed_transforms.DilateGT(dilation_factor=2),
        ivadomed_transforms.ROICrop2D(size=[48, 48]),
        ivadomed_transforms.ToTensor()
    ]
    train_transform = transforms.Compose(training_transform_list)

    train_lst = ['sub-test001']

    ds_train = loader.BidsDataset(PATH_BIDS,
                                  subject_lst=train_lst,
                                  target_suffix="_lesion-manual",
                                  roi_suffix="_seg-manual",
                                  contrast_lst=['T2w'],
                                  metadata_choice="without",
                                  contrast_balance={},
                                  slice_axis=2,
                                  transform=train_transform,
                                  multichannel=False,
                                  slice_filter_fn=SliceFilter(filter_empty_input=True, filter_empty_mask=True))

    ds_train.filter_roi(nb_nonzero_thr=10)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=1)

    if cuda_available:
        model.cuda()

    for i, batch in enumerate(train_loader):
        input_samples, gt_samples = batch["input"], batch["gt"]


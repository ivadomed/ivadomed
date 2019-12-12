import os
import numpy as np
import time
from random import randint

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, dataloader
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
DROPOUT = 0.4
BN = 0.1
SLICE_AXIS = 2
PATH_BIDS = 'testing_data/'
PATH_OUT = 'tmp/'

def test_undo(contrast='T2star'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        torch.cuda.set_device(GPU_NUMBER)
        print("using GPU number {}".format(GPU_NUMBER))

    validation_transform_list = [
        ivadomed_transforms.Resample(wspace=0.75, hspace=0.75),
        #ivadomed_transforms.CenterCrop2D(size=[40, 48]),
        ivadomed_transforms.ROICrop2D(size=[40, 48]),
        ivadomed_transforms.ToTensor(),
        ivadomed_transforms.NormalizeInstance()
    ]

    validation_transform_list_free = [
        ivadomed_transforms.ToTensor(),
        ivadomed_transforms.NormalizeInstance()
    ]

    val_transform = transforms.Compose(validation_transform_list)
    val_transform_undone = transforms.Compose(validation_transform_list_free)
    val_undo_transform = ivadomed_transforms.UndoCompose(val_transform)

    test_lst = ['sub-test001']

    ds_test = loader.BidsDataset(PATH_BIDS,
                                  subject_lst=test_lst,
                                  target_suffix="_lesion-manual",
                                  roi_suffix="_seg-manual",
                                  contrast_lst=[contrast],
                                  metadata_choice="contrast",
                                  contrast_balance={},
                                  slice_axis=SLICE_AXIS,
                                  transform=val_transform,
                                  multichannel=False,
                                  slice_filter_fn=SliceFilter(filter_empty_input=True,
                                                                filter_empty_mask=False))

    ds_test_noTrans = loader.BidsDataset(PATH_BIDS,
                                  subject_lst=test_lst,
                                  target_suffix="_lesion-manual",
                                  roi_suffix="_seg-manual",
                                  contrast_lst=[contrast],
                                  metadata_choice="contrast",
                                  contrast_balance={},
                                  slice_axis=SLICE_AXIS,
                                  transform=val_transform_undone,
                                  multichannel=False,
                                  slice_filter_fn=SliceFilter(filter_empty_input=True,
                                                                filter_empty_mask=False))

    test_loader = DataLoader(ds_test, batch_size=len(ds_test),
                             shuffle=False, pin_memory=True,
                             collate_fn=mt_datasets.mt_collate,
                             num_workers=1)
    batch_done = [t for i, t in enumerate(test_loader)][0]

    test_loader_noTrans = DataLoader(ds_test_noTrans, batch_size=len(ds_test_noTrans),
                             shuffle=False, pin_memory=True,
                             collate_fn=mt_datasets.mt_collate,
                             num_workers=1)
    batch_undone = [t for i, t in enumerate(test_loader_noTrans)][0]

    input_done, gt_done = batch_done["input"], batch_done["gt"]
    input_undone, gt_undone = batch_undone["input"], batch_undone["gt"]

    for smp_idx in range(len(batch_done['gt'])):
        # undo transformations
        rdict = {}
        for k in batch_done.keys():
            rdict[k] = batch_done[k][smp_idx]
        rdict_undo = val_undo_transform(rdict)

        before, after = np.array(gt_undone[smp_idx][0]), np.array(rdict_undo['gt'])
        after[after > 0] = 1.0
        print(before.shape == after.shape, np.sum(before) == np.sum(after), np.sum(before-after) == 0.0)
        print(before.shape, after.shape, np.sum(before), np.sum(after), np.sum(before-after), np.unique(before), np.unique(after))

        #fig = plt.figure(figsize=(20,10))
        #ax1 = fig.add_subplot(1,2,1)
        #im = ax1.imshow(np.array(before), cmap='gray')
        #ax2 = fig.add_subplot(1,2,2)
        #im = ax2.imshow(np.array(after), cmap='gray')
        #plt.savefig('tmpT/'+str(randint(0,1000))+'.png')
        #plt.close()

import os

import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as torch_transforms

import ivadomed.transforms as imed_transforms
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader
from ivadomed.utils import SliceFilter

cudnn.benchmark = True

GPU_NUMBER = 0
BATCH_SIZE = 8
DROPOUT = 0.4
BN = 0.1
N_EPOCHS = 10
INIT_LR = 0.01
PATH_BIDS = 'testing_data'
# PATH_BIDS = os.path.join(.., 'duke', 'sct_testing', 'large')
PATH_OUT = 'tmp_test_dilateGT'


def save_im_gt(im, gt, fname_out):
    plt.figure(figsize=(20, 10))

    i_zero, i_nonzero = np.where(gt == 0.0), np.nonzero(gt)
    img_jet = plt.cm.jet(plt.Normalize(vmin=0, vmax=1)(gt))
    img_jet[i_zero] = 0.0
    bkg_grey = plt.cm.binary_r(plt.Normalize(vmin=np.amin(im), vmax=np.amax(im))(im))
    img_out = np.copy(bkg_grey)
    img_out[i_nonzero] = img_jet[i_nonzero]

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(bkg_grey, interpolation='nearest', aspect='auto')

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(img_out, interpolation='nearest', aspect='auto')

    plt.savefig(fname_out, bbox_inches='tight', pad_inches=0)
    plt.close()


def test_dilateGT(dil_fac=0.25):
    training_transform_list = [
        imed_transforms.Resample(wspace=0.75, hspace=0.75),
        imed_transforms.DilateGT(dilation_factor=dil_fac),
        imed_transforms.ROICrop2D(size=[48, 48]),
        imed_transforms.NumpyToTensor()
    ]
    train_transform = torch_transforms.Compose(training_transform_list)

    train_lst = ['sub-test001']
    #    train_lst = ['sub-rennesMS056', 'sub-rennesMS057', 'sub-rennesMS058', 'sub-rennesMS059', 'sub-rennesMS060']
    #    train_lst = ['sub-nyuShepherd158', 'sub-nyuShepherd157', 'sub-nyuShepherd156', 'sub-nyuShepherd155']
    #    train_lst = ['sub-bwh070', 'sub-bwh071', 'sub-bwh072', 'sub-bwh073', 'sub-bwh074']

    ds_train = imed_loader.BidsDataset(PATH_BIDS,
                                       subject_lst=train_lst,
                                       target_suffix=["_lesion-manual"],
                                       roi_suffix="_seg-manual",
                                       contrast_lst=['T2w', 'acq-inf_T2star', 'acq-sup_T2star',
                                                     'acq-sup_T2w', 'acq-inf_T2w', 'acq-ax_T2w'],
                                       metadata_choice="without",
                                       contrast_balance={},
                                       slice_axis=2,
                                       transform=train_transform,
                                       multichannel=False,
                                       slice_filter_fn=SliceFilter(filter_empty_input=True, filter_empty_mask=True))
    print(len(ds_train))
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=1)

    # if not os.path.isdir(PATH_OUT):
    #    os.makedirs(PATH_OUT)

    for i, batch in enumerate(train_loader):
        input_samples, gt_samples = batch["input"], batch["gt"]
        for b_idx in range(len(batch['input'])):
            fname_out = os.path.join(PATH_OUT, 'im_' + str(i).zfill(2) +
                                     '_' + str(b_idx).zfill(2) + '.png')
    #       save_im_gt(np.array(input_samples[b_idx, 0]),
    #                   np.array(gt_samples[b_idx, 0]),
    #                   fname_out)


test_dilateGT()

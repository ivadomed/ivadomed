##############################################################
#
# This script tests the data augmentation DilateGT (ivadomed/utils).
#
# Usage: python dev/test_dilateGT.py -o <ofolder>
#
# Example: python dev/test_dilateGT.py -o test_dilate
#
# Contributors: charley
# Last modified: 24-10-2019
#
##############################################################

import os
import numpy as np
import argparse
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from ivadomed.utils import DilateGT
from medicaltorch.datasets import MRI2DSegmentationDataset
from medicaltorch import transforms as mt_transforms


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", help="Quality control output folder.")

    return parser


#def save_sample(arr, fname_out):
#    plt.figure()
#    plt.subplot(1, 1, 1)
#    plt.axis("off")

#    plt.imshow(arr, interpolation='nearest', aspect='auto', cmap='jet', vmin=0, vmax=1)

#    plt.savefig(fname_out, bbox_inches='tight', pad_inches=0)
#    plt.close()


def save_im_gt(im, gt, fname_out):
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.axis("off")

    i_zero, i_nonzero = np.where(gt==0.0), np.nonzero(gt)
    img_jet = plt.cm.jet(plt.Normalize(vmin=0, vmax=1)(gt))
    img_jet[i_zero] = 0.0
    bkg_grey = plt.cm.binary_r(plt.Normalize(vmin=np.amin(im), vmax=np.amax(im))(im))
    img_out = np.copy(bkg_grey)
    img_out[i_nonzero] = img_jet[i_nonzero]

    plt.imshow(img_out, interpolation='nearest', aspect='auto')

    plt.savefig(fname_out, bbox_inches='tight', pad_inches=0)
    plt.close()


#def test_output_shape(in_np, out_np):
#    if out_np.shape != in_np.shape:
#        print('\nDifferent shape: {} vs. {}'.format(','.join([str(s) for s in out_np.shape]),
#                                                    ','.join([str(ss) for ss in in_np.shape])))
#    else:
#        print('\nCorrect output tensor size!')


#def test_gt_preserved(in_np, out_np):
#    out_np_ones = (out_np == 1).astype(np.int)
#    if not np.array_equal(in_np.astype(np.int), out_np_ones):
#        print('\nError: GT has not been preserved!')
#    else:
#        print('\nGT has been preserved.')


#def qc_dummy(in_np, out_np, ofolder, prefix):
#    for ii in range(in_np.shape[0]):
#        save_sample(in_np[ii,0], os.path.join(ofolder, prefix + str(ii).zfill(2)+'_in.png'))
#        save_sample(out_np[ii,0], os.path.join(ofolder, prefix + str(ii).zfill(2)+'_out.png'))
#    print('\nQuality control saved in {}'.format(ofolder))


def qc(im_lst, gt_lst, ofolder, suffix):
    for idx, im, gt in zip(range(len(im_lst)), im_lst, gt_lst):
        save_im_gt(np.array(im), np.array(gt), os.path.join(ofolder, str(idx).zfill(2)+suffix+'.png'))
    print('\nQuality control saved in {}'.format(ofolder))


def true_data_test(fname_im, fname_gt, z_lst, ofolder, dilation_factor):
    # init
    transform = DilateGT(dilation_factor)

    # transforms
    transform_lst = transforms.Compose([
                    mt_transforms.Resample(wspace=0.75, hspace=0.75),
                    mt_transforms.CenterCrop2D((128, 128)),
                    DilateGT(dilation_factor)])

    # data
    ds = MRI2DSegmentationDataset([(fname_im, fname_gt)], transform=transform_lst)
    im_lst, gt_lst = [], []
    for z in z_lst:
        ds_dct = ds.__getitem__(z)
        im_lst.append(ds_dct['input'])
        gt_lst.append(ds_dct['gt'])

    qc(im_lst, gt_lst, ofolder, '_dil')

    # transforms
    transform_lst = transforms.Compose([
                    mt_transforms.Resample(wspace=0.75, hspace=0.75),
                    mt_transforms.CenterCrop2D((128, 128))])

    # data
    ds = MRI2DSegmentationDataset([(fname_im, fname_gt)], transform=transform_lst)
    im_lst, gt_lst = [], []
    for z in z_lst:
        ds_dct = ds.__getitem__(z)
        im_lst.append(ds_dct['input'])
        gt_lst.append(ds_dct['gt'])

    qc(im_lst, gt_lst, ofolder, '_no-dil')


#def dummy_test(ofolder, dilation_factor=0.3):
#    # init
#    transform = DilateGT(dilation_factor)

#    # dummy data
#    a=torch.zeros([16, 1, 128, 128], dtype=torch.int32)
#    a[8,0,25:105,50:100] = 1
#    a[10,0,10:20,10:20] = 1
#    a[10,0,100:120,80:120] = 1
#    sample = {'gt': a}
#    gt_in_np = a.numpy()

#    # run data augmentation
#    sample_out = transform.__call__(sample)
#    gt_out_np = sample_out['gt'].numpy()

#    # run tests
#    test_output_shape(gt_in_np, gt_out_np)
#    test_gt_preserved(gt_in_np, gt_out_np)
#    qc(gt_in_np, gt_out_np, ofolder, 'dummy_')


def run_tests(args):
    ofolder = args.o
    if not os.path.isdir(ofolder):
        os.makedirs(ofolder)
    else:
        print('\nWarning: ofolder already exists')

    # on dummy data
    #dummy_test(ofolder, 0.3)

    # on true data
    fname_im = '../duke/sct_testing/large/sub-bwh035/anat/sub-bwh035_acq-ax_T2w.nii.gz'
    fname_gt = '../duke/sct_testing/large/derivatives/labels/sub-bwh035/anat/sub-bwh035_acq-ax_T2w_lesion-manual.nii.gz'
    z_lst = [10, 29, 30, 31, 32]
    true_data_test(fname_im, fname_gt, z_lst, ofolder, 0.5)

if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    run_tests(arguments)

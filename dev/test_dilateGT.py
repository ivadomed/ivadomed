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
import matplotlib.pyplot as plt

from ivadomed.utils import DilateGT


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", help="Quality control output folder.")

    return parser


def save_sample(arr, fname_out):
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.axis("off")

    plt.imshow(arr, interpolation='nearest', aspect='auto', cmap='jet', vmin=0, vmax=1)

    plt.savefig(fname_out, bbox_inches='tight', pad_inches=0)
    plt.close()


def run_test(args):
    ofolder = args.o
    if not os.path.isdir(ofolder):
        os.makedirs(ofolder)
    else:
        print('\nWarning: ofolder already exists')

    # init
    transform = DilateGT(0.5)

    # dummy data
    a=torch.zeros([16, 1, 128, 128], dtype=torch.int32)
    a[8,0,25:105,50:100] = 1
    a[10,0,10:20,10:20] = 1
    a[10,0,100:120,80:120] = 1
    sample = {'gt': a}
    gt_in_np = a.numpy()

    # run data augmentation
    sample_out = transform.__call__(sample)
    gt_out_np = sample_out['gt'].numpy()

    # test output shape
    if gt_out_np.shape != gt_in_np.shape:
        print('\nDifferent shape: {} vs. {}'.format(','.join([str(s) for s in gt_out_np.shape]),
                                                    ','.join([str(ss) for ss in gt_in_np.shape])))
    else:
        print('\nCorrect output tensor size!')

    # test if GT has been preserved
    gt_out_np_ones = (gt_out_np == 1).astype(np.int)
    if not np.array_equal(gt_in_np.astype(np.int), gt_out_np_ones):
        print('\nError: GT has not been preserved!')
    else:
        print('\nGT has been preserved.')

    # save quality control
    for ii in range(gt_in_np.shape[0]):
        save_sample(gt_in_np[ii,0], os.path.join(ofolder, str(ii).zfill(2)+'_in.png'))
        save_sample(gt_out_np[ii,0], os.path.join(ofolder, str(ii).zfill(2)+'_out.png'))
    print('\nQuality control saved in {}'.format(ofolder))

if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    run_test(arguments)

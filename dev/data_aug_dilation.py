#!/usr/bin/env python
#
# This script is used to develop a data augmentation method of mask dilation
#
# Example: python data_aug_dilation.py t2_seg.nii.gz
#

import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from spinalcordtoolbox.image import Image
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes, binary_closing


def save_sample(img, fname_out):
    plt.figure()

    plt.subplot(1, 1, 1)
    plt.axis("off")

    plt.imshow(img, interpolation='nearest', aspect='auto', cmap='jet', vmin=0, vmax=1)

    plt.savefig(fname_out, bbox_inches='tight', pad_inches=0)
    plt.close()


def dilate_mask(mask, nb_dilation_it=3):
    # values of the voxels added to the input mask
    soft_label_values = [x / (nb_dilation_it+1) for x in range(nb_dilation_it, 0, -1)]

    # dilation
    mask_bin, mask_soft = mask.astype(np.int), mask.astype(np.float)
    for soft_label in soft_label_values:
        # binary dilation with 1 iteration
        mask_dilated = binary_dilation(mask_bin, iterations=1)

        # isolate new voxels, i.e. the ones from the dilation
        new_voxels = np.logical_xor(mask_dilated, mask_bin).astype(np.int)

        # assign a soft value (]0, 1[) to the new voxels
        soft_new_voxels = soft_label * new_voxels

        # add the new voxels to the input mask
        mask_soft += soft_new_voxels
        mask_bin = (mask_soft > 0).astype(np.int)

    # save the mask after dilation, used later during post-processing
    mask_after_dilation = np.copy(mask_soft)

    # coordinates of the new voxels, i.e. the ones from the dilation
    new_voxels_xx, new_voxels_yy = np.where(np.logical_xor(mask_bin, mask))
    nb_new_voxels = new_voxels_xx.shape[0]

    # ratio of voxels added to the input mask from the dilated mask
    new_voxel_ratio = random.random()
    # randomly select new voxel indexes to remove
    idx_to_remove = random.sample(range(nb_new_voxels),
                                    int(round(nb_new_voxels * (1 - new_voxel_ratio))))
    # set to zero the here-above randomly selected new voxels
    mask_soft[new_voxels_xx[idx_to_remove], new_voxels_yy[idx_to_remove]] = 0
    mask_bin = (mask_soft > 0).astype(np.int)

    # remove new object that are not connected to the input mask
    mask_labeled, labels_nb = label(mask_bin)
    connected_to_input_mask = mask_labeled * mask
    for label_value in range(1, labels_nb+1):
        if np.sum(connected_to_input_mask == label_value) == 0:
            mask_soft[mask_labeled == label_value] = 0

    # fill binary holes
    mask_bin = binary_fill_holes((mask_soft > 0).astype(np.int))
    # binary closing
    mask_bin = binary_closing(mask_bin.astype(np.int))
    # recover the soft-value assigned to the filled-holes
    mask_out = mask_bin * mask_after_dilation

    # return mask

def run_main(mask):

    mask_im = Image(mask)  # assume RPI orientation
    mask_data = mask_im.data[:, :, random.sample(range(mask_im.dim[2]), 1)[0]]

    for i in range(10):
        dilate_mask((mask_data > 0).astype(np.int), nb_dilation_it=2)

if __name__ == '__main__':
    input_mask, ofolder = sys.argv[1:]
    run_main(input_mask[0])

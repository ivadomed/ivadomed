import math
import random
import numpy as np
from scipy.ndimage import label, binary_dilation, binary_fill_holes, binary_closing
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible


class DilateGT(ImedTransform):
    """Randomly dilate a ground-truth tensor.

    .. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/technical_features/dilate-gt.png
        :width: 500px
        :align: center

    Args:
        dilation_factor (float): Controls the number of dilation iterations. For each individual lesion, the number of
        dilation iterations is computed as follows:
            nb_it = int(round(dilation_factor * sqrt(lesion_area)))
        If dilation_factor <= 0, then no dilation will be performed.
    """

    def __init__(self, dilation_factor):
        self.dil_factor = dilation_factor

    @staticmethod
    def dilate_lesion(arr_bin, arr_soft, label_values):
        for lb in label_values:
            # binary dilation with 1 iteration
            arr_dilated = binary_dilation(arr_bin, iterations=1)

            # isolate new voxels, i.e. the ones from the dilation
            new_voxels = np.logical_xor(arr_dilated, arr_bin).astype(int)

            # assign a soft value (]0, 1[) to the new voxels
            soft_new_voxels = lb * new_voxels

            # add the new voxels to the input mask
            arr_soft += soft_new_voxels
            arr_bin = (arr_soft > 0).astype(int)

        return arr_bin, arr_soft

    def dilate_arr(self, arr, dil_factor):
        # identify each object
        arr_labeled, lb_nb = label(arr.astype(int))

        # loop across each object
        arr_bin_lst, arr_soft_lst = [], []
        for obj_idx in range(1, lb_nb + 1):
            arr_bin_obj = (arr_labeled == obj_idx).astype(int)
            arr_soft_obj = np.copy(arr_bin_obj).astype(float)
            # compute the number of dilation iterations depending on the size of the lesion
            nb_it = int(round(dil_factor * math.sqrt(arr_bin_obj.sum())))
            # values of the voxels added to the input mask
            soft_label_values = [x / (nb_it + 1) for x in range(nb_it, 0, -1)]
            # dilate lesion
            arr_bin_dil, arr_soft_dil = self.dilate_lesion(arr_bin_obj, arr_soft_obj, soft_label_values)
            arr_bin_lst.append(arr_bin_dil)
            arr_soft_lst.append(arr_soft_dil)

        # sum dilated objects
        arr_bin_idx = np.sum(np.array(arr_bin_lst), axis=0)
        arr_soft_idx = np.sum(np.array(arr_soft_lst), axis=0)
        # clip values in case dilated voxels overlap
        arr_bin_clip, arr_soft_clip = np.clip(arr_bin_idx, 0, 1), np.clip(arr_soft_idx, 0.0, 1.0)

        return arr_soft_clip.astype(float), arr_bin_clip.astype(int)

    @staticmethod
    def random_holes(arr_in, arr_soft, arr_bin):
        arr_soft_out = np.copy(arr_soft)

        # coordinates of the new voxels, i.e. the ones from the dilation
        new_voxels_xx, new_voxels_yy, new_voxels_zz = np.where(np.logical_xor(arr_bin, arr_in))
        nb_new_voxels = new_voxels_xx.shape[0]

        # ratio of voxels added to the input mask from the dilated mask
        new_voxel_ratio = random.random()
        # randomly select new voxel indexes to remove
        idx_to_remove = random.sample(range(nb_new_voxels),
                                      int(round(nb_new_voxels * (1 - new_voxel_ratio))))

        # set to zero the here-above randomly selected new voxels
        arr_soft_out[new_voxels_xx[idx_to_remove],
                     new_voxels_yy[idx_to_remove],
                     new_voxels_zz[idx_to_remove]] = 0.0
        arr_bin_out = (arr_soft_out > 0).astype(int)

        return arr_soft_out, arr_bin_out

    @staticmethod
    def post_processing(arr_in, arr_soft, arr_bin, arr_dil):
        # remove new object that are not connected to the input mask
        arr_labeled, lb_nb = label(arr_bin)

        connected_to_in = arr_labeled * arr_in
        for lb in range(1, lb_nb + 1):
            if np.sum(connected_to_in == lb) == 0:
                arr_soft[arr_labeled == lb] = 0

        struct = np.ones((3, 3, 1) if arr_soft.shape[2] == 1 else (3, 3, 3))
        # binary closing
        arr_bin_closed = binary_closing((arr_soft > 0).astype(int), structure=struct)
        # fill binary holes
        arr_bin_filled = binary_fill_holes(arr_bin_closed)

        # recover the soft-value assigned to the filled-holes
        arr_soft_out = arr_bin_filled * arr_dil

        return arr_soft_out

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # binarize for processing
        gt_data_np = (sample > 0.5).astype(np.int_)

        if self.dil_factor > 0 and np.sum(sample):
            # dilation
            gt_dil, gt_dil_bin = self.dilate_arr(gt_data_np, self.dil_factor)

            # random holes in dilated area
            # gt_holes, gt_holes_bin = self.random_holes(gt_data_np, gt_dil, gt_dil_bin)

            # post-processing
            # gt_pp = self.post_processing(gt_data_np, gt_holes, gt_holes_bin, gt_dil)

            # return gt_pp.astype(np.float32), metadata
            return gt_dil.astype(np.float32), metadata

        else:
            return sample, metadata


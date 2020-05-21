import json
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from scipy.ndimage import label, generate_binary_structure
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from ivadomed import metrics as imed_metrics
from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed.loader import utils as imed_loaded_utils, loader as imed_loader

# labels of paint_objects method
TP_COLOUR = 1
FP_COLOUR = 2
FN_COLOUR = 3

AXIS_DCT = {'sagittal': 0, 'coronal': 1, 'axial': 2}


class Evaluation3DMetrics(object):

    def __init__(self, data_pred, data_gt, dim_lst, params={}):

        self.data_pred = data_pred
        if len(self.data_pred.shape) == 3:
            self.data_pred = np.expand_dims(self.data_pred, -1)

        self.data_gt = data_gt
        if len(self.data_gt.shape) == 3:
            self.data_gt = np.expand_dims(self.data_gt, -1)

        h, w, d, self.n_classes = self.data_gt.shape
        self.px, self.py, self.pz = dim_lst

        self.bin_struct = generate_binary_structure(3, 2)  # 18-connectivity

        # Remove small objects
        if "removeSmall" in params:
            size_min = params['removeSmall']['thr']
            if params['removeSmall']['unit'] == 'vox':
                self.size_min = size_min
            elif params['removeSmall']['unit'] == 'mm3':
                self.size_min = np.round(size_min / (self.px * self.py * self.pz))
            else:
                print('Please choose a different unit for removeSmall. Chocies: vox or mm3')
                exit()

            for idx in range(self.n_classes):
                self.data_pred[..., idx] = self.remove_small_objects(data=self.data_pred[..., idx])
                self.data_gt[..., idx] = self.remove_small_objects(data=self.data_gt[..., idx])
        else:
            self.size_min = 0

        if "targetSize" in params:
            self.size_rng_lst, self.size_suffix_lst = \
                self._get_size_ranges(thr_lst=params["targetSize"]["thr"],
                                      unit=params["targetSize"]["unit"])
            self.label_size_lst = []
            self.data_gt_per_size = np.zeros(self.data_gt.shape)
            self.data_pred_per_size = np.zeros(self.data_gt.shape)
            for idx in range(self.n_classes):
                self.data_gt_per_size[..., idx] = self.label_per_size(self.data_gt[..., idx])
                label_gt_size_lst = list(set(self.data_gt_per_size[np.nonzero(self.data_gt_per_size)]))
                self.data_pred_per_size[..., idx] = self.label_per_size(self.data_pred[..., idx])
                label_pred_size_lst = list(set(self.data_pred_per_size[np.nonzero(self.data_pred_per_size)]))
                self.label_size_lst.append([label_gt_size_lst + label_pred_size_lst,
                                            ['gt'] * len(label_gt_size_lst) + ['pred'] * len(label_pred_size_lst)])

        else:
            self.label_size_lst = [[[], []] * self.n_classes]

        # 18-connected components
        self.data_pred_label = np.zeros((h, w, d, self.n_classes), dtype='u1')
        self.data_gt_label = np.zeros((h, w, d, self.n_classes), dtype='u1')
        self.n_pred = [None] * self.n_classes
        self.n_gt = [None] * self.n_classes
        for idx in range(self.n_classes):
            self.data_pred_label[..., idx], self.n_pred[idx] = label(self.data_pred[..., idx],
                                                                     structure=self.bin_struct)
            self.data_gt_label[..., idx], self.n_gt[idx] = label(self.data_gt[..., idx],
                                                                 structure=self.bin_struct)

        # painted data, object wise
        self.data_painted = np.copy(self.data_pred)

        # overlap_vox is used to define the object-wise TP, FP, FN
        if "overlap" in params:
            if params["overlap"]["unit"] == 'vox':
                self.overlap_vox = params["overlap"]["thr"]
            elif params["overlap"]["unit"] == 'mm3':
                self.overlap_vox = np.round(params["overlap"]["thr"] / (self.px * self.py * self.pz))
            elif params["overlap"]["unit"] == 'percent':  # percentage of the GT object
                self.overlap_percent = params["overlap"]["thr"]
                self.overlap_vox = None
        else:
            self.overlap_vox = 3

    def remove_small_objects(self, data):
        data_label, n = label(data,
                              structure=self.bin_struct)

        for idx in range(1, n + 1):
            data_idx = (data_label == idx).astype(np.int)
            n_nonzero = np.count_nonzero(data_idx)

            if n_nonzero < self.size_min:
                data[data_label == idx] = 0

        return data

    def _get_size_ranges(self, thr_lst, unit):
        assert unit in ['vox', 'mm3']

        rng_lst, suffix_lst = [], []
        for i, thr in enumerate(thr_lst):
            if i == 0:
                thr_low = self.size_min
            else:
                thr_low = thr_lst[i - 1] + 1

            thr_high = thr

            if unit == 'mm3':
                thr_low = np.round(thr_low / (self.px * self.py * self.pz))
                thr_high = np.round(thr_high / (self.px * self.py * self.pz))

            rng_lst.append([thr_low, thr_high])

            suffix_lst.append('_' + str(thr_low) + '-' + str(thr_high) + unit)

        # last subgroup
        thr_low = thr_lst[i] + 1
        if unit == 'mm3':
            thr_low = np.round(thr_low / (self.px * self.py * self.pz))
        thr_high = np.inf
        rng_lst.append([thr_low, thr_high])
        suffix_lst.append('_' + str(thr_low) + '-INF' + unit)

        return rng_lst, suffix_lst

    def label_per_size(self, data):
        data_label, n = label(data,
                              structure=self.bin_struct)
        data_out = np.zeros(data.shape)

        for idx in range(1, n + 1):
            data_idx = (data_label == idx).astype(np.int)
            n_nonzero = np.count_nonzero(data_idx)

            for idx_size, rng in enumerate(self.size_rng_lst):
                if n_nonzero >= rng[0] and n_nonzero <= rng[1]:
                    data_out[np.nonzero(data_idx)] = idx_size + 1

        return data_out.astype(np.int)

    def get_vol(self, data):
        vol = np.sum(data)
        vol *= self.px * self.py * self.pz
        return vol

    def get_rvd(self):
        """Relative volume difference."""
        vol_gt = self.get_vol(self.data_gt)
        vol_pred = self.get_vol(self.data_pred)

        if vol_gt == 0.0:
            return np.nan

        rvd = (vol_gt - vol_pred) * 100.
        rvd /= vol_gt

        return rvd

    def get_avd(self):
        """Absolute volume difference."""
        return abs(self.get_rvd())

    def _get_ltp_lfn(self, label_size, class_idx=0):
        """Number of true positive and false negative lesion.

            Note1: if two lesion_pred overlap with the current lesion_gt,
                then only one detection is counted.
        """
        ltp, lfn, n_obj = 0, 0, 0

        for idx in range(1, self.n_gt[class_idx] + 1):
            data_gt_idx = (self.data_gt_label[..., class_idx] == idx).astype(np.int)
            overlap = (data_gt_idx * self.data_pred[..., class_idx]).astype(np.int)

            # if label_size is None, then we look at all object sizes
            # we check if the currrent object belongs to the current size range
            if label_size is None or \
                    np.max(self.data_gt_per_size[..., class_idx][np.nonzero(data_gt_idx)]) == label_size:

                if self.overlap_vox is None:
                    overlap_vox = np.round(np.count_nonzero(data_gt_idx) * self.overlap_percent / 100.)
                else:
                    overlap_vox = self.overlap_vox

                if np.count_nonzero(overlap) >= overlap_vox:
                    ltp += 1

                else:
                    lfn += 1

                    if label_size is None:  # painting is done while considering all objects
                        self.data_painted[..., class_idx][self.data_gt_label[..., class_idx] == idx] = FN_COLOUR

                n_obj += 1

        return ltp, lfn, n_obj

    def _get_lfp(self, label_size, class_idx=0):
        """Number of false positive lesion."""
        lfp = 0
        for idx in range(1, self.n_pred[class_idx] + 1):
            data_pred_idx = (self.data_pred_label[..., class_idx] == idx).astype(np.int)
            overlap = (data_pred_idx * self.data_gt[..., class_idx]).astype(np.int)

            label_gt = np.max(data_pred_idx * self.data_gt_label[..., class_idx])
            data_gt_idx = (self.data_gt_label[..., class_idx] == label_gt).astype(np.int)
            # if label_size is None, then we look at all object sizes
            # we check if the current object belongs to the current size range

            if label_size is None or \
                    np.max(self.data_pred_per_size[..., class_idx][np.nonzero(data_gt_idx)]) == label_size:

                if self.overlap_vox is None:
                    overlap_thr = np.round(np.count_nonzero(data_gt_idx) * self.overlap_percent / 100.)
                else:
                    overlap_thr = self.overlap_vox

                if np.count_nonzero(overlap) < overlap_thr:
                    lfp += 1
                    if label_size is None:  # painting is done while considering all objects
                        self.data_painted[..., class_idx][self.data_pred_label[..., class_idx] == idx] = FP_COLOUR
                else:
                    if label_size is None:  # painting is done while considering all objects
                        self.data_painted[..., class_idx][self.data_pred_label[..., class_idx] == idx] = TP_COLOUR

        return lfp

    def get_ltpr(self, label_size=None, class_idx=0):
        """Lesion True Positive Rate / Recall / Sensitivity.

            Note: computed only if n_obj >= 1.
        """
        ltp, lfn, n_obj = self._get_ltp_lfn(label_size, class_idx)

        denom = ltp + lfn
        if denom == 0 or n_obj == 0:
            return np.nan, n_obj

        return ltp * 100. / denom, n_obj

    def get_lfdr(self, label_size=None, class_idx=0):
        """Lesion False Detection Rate / 1 - Precision.

            Note: computed only if n_obj >= 1.
        """
        ltp, _, n_obj = self._get_ltp_lfn(label_size, class_idx)
        lfp = self._get_lfp(label_size, class_idx)

        denom = ltp + lfp
        if denom == 0 or n_obj == 0:
            return np.nan

        return lfp * 100. / denom

    def run_eval(self):
        dct = {}

        for n in range(self.n_classes):
            dct['vol_pred_class' + str(n)] = self.get_vol(self.data_pred)
            dct['vol_gt_class' + str(n)] = self.get_vol(self.data_gt)
            dct['rvd_class' + str(n)], dct['avd_class' + str(n)] = self.get_rvd(), self.get_avd()
            dct['dice_class' + str(n)] = imed_metrics.dice_score(self.data_gt[..., n], self.data_pred[..., n]) * 100.
            dct['recall_class' + str(n)] = imed_metrics.recall_score(self.data_pred, self.data_gt, err_value=np.nan)
            dct['precision_class' + str(n)] = imed_metrics.precision_score(self.data_pred, self.data_gt,
                                                                           err_value=np.nan)
            dct['specificity_class' + str(n)] = imed_metrics.specificity_score(self.data_pred, self.data_gt,
                                                                               err_value=np.nan)
            dct['n_pred_class' + str(n)], dct['n_gt_class' + str(n)] = self.n_pred[n], self.n_gt[n]
            dct['ltpr_class' + str(n)], _ = self.get_ltpr()
            dct['lfdr_class' + str(n)] = self.get_lfdr()

            for lb_size, gt_pred in zip(self.label_size_lst[n][0], self.label_size_lst[n][1]):
                suffix = self.size_suffix_lst[int(lb_size) - 1]

                if gt_pred == 'gt':
                    dct['ltpr' + suffix + "_class" + str(n)], dct['n' + suffix] = self.get_ltpr(label_size=lb_size,
                                                                                                class_idx=n)
                else:  # gt_pred == 'pred'
                    dct['lfdr' + suffix + "_class" + str(n)] = self.get_lfdr(label_size=lb_size, class_idx=n)

        if self.n_classes == 1:
            self.data_painted = np.squeeze(self.data_painted, axis=-1)

        return dct, self.data_painted


def pred_to_nib(data_lst, z_lst, fname_ref, fname_out, slice_axis, debug=False, kernel_dim='2d', bin_thr=0.5,
                discard_noise=True):
    """Convert the NN predictions as nibabel object.

    Based on the header of fname_ref image, it creates a nibabel object from the NN predictions (data_lst),
    given the slice indexes (z_lst) along slice_axis.

    Args:
        data_lst (list of np arrays): predictions, either 2D slices either 3D patches.
        z_lst (list of ints): slice indexes to reconstruct a 3D volume.
        fname_ref (string): Filename of the input image: its header is copied to the output nibabel object.
        fname_out (string): If not None, then the generated nibabel object is saved with this filename.
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
        debug (bool): Display additional info used for debugging.
        kernel_dim (string): Indicates the number of dimensions of the extracted patches. Choices: '2d', '3d'.
        bin_thr (float): If positive, then the segmentation is binarised with this given threshold.
    Returns:
        NibabelObject: Object containing the NN prediction.

    """
    # Load reference nibabel object
    nib_ref = nib.load(fname_ref)
    nib_ref_can = nib.as_closest_canonical(nib_ref)

    if kernel_dim == '2d':
        # complete missing z with zeros
        tmp_lst = []
        for z in range(nib_ref_can.header.get_data_shape()[slice_axis]):
            if not z in z_lst:
                tmp_lst.append(np.zeros(data_lst[0].shape))
            else:
                tmp_lst.append(data_lst[z_lst.index(z)])

        if debug:
            print("Len {}".format(len(tmp_lst)))
            for arr in tmp_lst:
                print("Shape element lst {}".format(arr.shape))

        # create data and stack on depth dimension
        arr = np.stack(tmp_lst, axis=-1)

        # If only one channel then return 3D arr
        if arr.shape[0] == 1:
            arr = arr[0, :]

        # Reorient data
        arr_pred_ref_space = reorient_image(arr, slice_axis, nib_ref, nib_ref_can).astype('float32')

    else:
        arr_pred_ref_space = data_lst[0]
        n_channel = arr_pred_ref_space.shape[0]
        oriented_volumes = []
        if len(arr_pred_ref_space.shape) == 4:
            for i in range(n_channel):
                oriented_volumes.append(reorient_image(arr_pred_ref_space[i,], slice_axis, nib_ref, nib_ref_can))
            # transpose to locate the channel dimension at the end to properly see image on viewer
            arr_pred_ref_space = np.asarray(oriented_volumes).transpose((1, 2, 3, 0))

    # If only one channel then return 3D arr
    if arr_pred_ref_space.shape[0] == 1:
        arr_pred_ref_space = arr_pred_ref_space[0, :]

    if bin_thr >= 0:
        arr_pred_ref_space = imed_postpro.threshold_predictions(arr_pred_ref_space, thr=bin_thr)
    elif discard_noise:  # discard noise
        arr_pred_ref_space[arr_pred_ref_space <= 1e-1] = 0

    # create nibabel object
    nib_pred = nib.Nifti1Image(arr_pred_ref_space, nib_ref.affine)

    # save as nifti file
    if fname_out is not None:
        nib.save(nib_pred, fname_out)

    return nib_pred


def run_uncertainty(ifolder):
    # list subj_acq prefixes
    subj_acq_lst = [f.split('_pred')[0] for f in os.listdir(ifolder)
                    if f.endswith('.nii.gz') and '_pred' in f]
    # remove duplicates
    subj_acq_lst = list(set(subj_acq_lst))
    # keep only the images where unc has not been computed yet
    subj_acq_lst = [f for f in subj_acq_lst if not os.path.isfile(
        os.path.join(ifolder, f + '_unc-cv.nii.gz'))]

    # loop across subj_acq
    for subj_acq in tqdm(subj_acq_lst, desc="Uncertainty Computation"):
        # hard segmentation from MC samples
        fname_pred = os.path.join(ifolder, subj_acq + '_pred.nii.gz')
        # fname for soft segmentation from MC simulations
        fname_soft = os.path.join(ifolder, subj_acq + '_soft.nii.gz')
        # find Monte Carlo simulations
        fname_pred_lst = [os.path.join(ifolder, f)
                          for f in os.listdir(ifolder) if subj_acq + '_pred_' in f]

        # if final segmentation from Monte Carlo simulations has not been generated yet
        if not os.path.isfile(fname_pred) or not os.path.isfile(fname_soft):
            # threshold used for the hard segmentation
            thr = 1. / len(fname_pred_lst)  # 1 for all voxels where at least on MC sample predicted 1
            # average then argmax
            combine_predictions(fname_pred_lst, fname_pred, fname_soft, thr=thr)

        fname_unc_vox = os.path.join(ifolder, subj_acq + '_unc-vox.nii.gz')
        if not os.path.isfile(fname_unc_vox):
            # compute voxel-wise uncertainty map
            voxelWise_uncertainty(fname_pred_lst, fname_unc_vox)

        fname_unc_struct = os.path.join(ifolder, subj_acq + '_unc.nii.gz')
        if not os.path.isfile(os.path.join(ifolder, subj_acq + '_unc-cv.nii.gz')):
            # compute structure-wise uncertainty
            structureWise_uncertainty(fname_pred_lst, fname_pred, fname_unc_vox, fname_unc_struct)


def combine_predictions(fname_lst, fname_hard, fname_prob, thr=0.5):
    """
    Combine predictions from Monte Carlo simulations
    by applying:
        (1) a mean (saved as fname_prob)
        (2) then argmax operation (saved as fname_hard).
    """
    # collect all MC simulations
    data_lst = []
    for fname in fname_lst:
        nib_im = nib.load(fname)
        data_lst.append(nib_im.get_fdata())

    # average over all the MC simulations
    data_prob = np.mean(np.array(data_lst), axis=0)
    # save prob segmentation
    nib_prob = nib.Nifti1Image(data_prob, nib_im.affine)
    nib.save(nib_prob, fname_prob)

    # argmax operator
    # TODO: adapt for multi-label pred
    data_hard = imed_postpro.threshold_predictions(data_prob, thr=thr).astype(np.uint8)
    # save hard segmentation
    nib_hard = nib.Nifti1Image(data_hard, nib_im.affine)
    nib.save(nib_hard, fname_hard)


def voxelWise_uncertainty(fname_lst, fname_out, eps=1e-5):
    """
    Voxel-wise uncertainty is estimated as entropy over all
    N MC probability maps, and saved in fname_out.
    """
    # collect all MC simulations
    data_lst = []
    for fname in fname_lst:
        nib_im = nib.load(fname)
        data_lst.append(nib_im.get_fdata())

    # entropy
    unc = np.repeat(np.expand_dims(np.array(data_lst), -1), 2, -1)  # n_it, x, y, z, 2
    unc[..., 0] = 1 - unc[..., 1]
    unc = -np.sum(np.mean(unc, 0) * np.log(np.mean(unc, 0) + eps), -1)

    # save uncertainty map
    nib_unc = nib.Nifti1Image(unc, nib_im.affine)
    nib.save(nib_unc, fname_out)


def structureWise_uncertainty(fname_lst, fname_hard, fname_unc_vox, fname_out):
    """
    Structure-wise uncertainty from N MC probability maps (fname_lst)
    and saved in fname_out with the following suffixes:
       - '-cv.nii.gz': coefficient of variation
       - '-iou.nii.gz': intersection over union
       - '-avgUnc.nii.gz': average voxel-wise uncertainty within the structure.
    """
    # load hard segmentation and label it
    nib_hard = nib.load(fname_hard)
    data_hard = nib_hard.get_fdata()
    bin_struct = generate_binary_structure(3, 2)  # 18-connectivity
    data_hard_l, n_l = label(data_hard, structure=bin_struct)

    # load uncertainty map
    nib_uncVox = nib.load(fname_unc_vox)
    data_uncVox = nib_uncVox.get_fdata()
    del nib_uncVox

    # init output arrays
    data_iou, data_cv, data_avgUnc = np.zeros(data_hard.shape), np.zeros(data_hard.shape), np.zeros(data_hard.shape)

    # load all MC simulations and label them
    data_lst, data_l_lst = [], []
    for fname in fname_lst:
        nib_im = nib.load(fname)
        data_im = nib_im.get_fdata()
        data_lst.append(data_im)
        data_im_l, _ = label(data_im, structure=bin_struct)
        data_l_lst.append(data_im_l)
        del nib_im

    # loop across all structures of data_hard_l
    for i_l in range(1, n_l + 1):
        # select the current structure, remaining voxels are set to zero
        data_i_l = (data_hard_l == i_l).astype(np.int)

        # select the current structure in each MC sample
        # and store it in data_mc_i_l_lst
        data_mc_i_l_lst = []
        # loop across MC samples
        for i_mc in range(len(data_lst)):
            # find the structure of interest in the current MC sample
            data_i_inter = data_i_l * data_l_lst[i_mc]
            i_mc_l = np.max(data_i_inter)

            if i_mc_l > 0:
                # keep only the unc voxels of the structure of interest
                data_mc_i_l = np.copy(data_lst[i_mc])
                data_mc_i_l[data_l_lst[i_mc] != i_mc_l] = 0.
            else:  # no structure in this sample
                data_mc_i_l = np.zeros(data_lst[i_mc].shape)
            data_mc_i_l_lst.append(data_mc_i_l)

        # compute IoU over all the N MC samples for a specific structure
        intersection = np.logical_and(data_mc_i_l_lst[0].astype(np.bool),
                                      data_mc_i_l_lst[1].astype(np.bool))
        union = np.logical_or(data_mc_i_l_lst[0].astype(np.bool),
                              data_mc_i_l_lst[1].astype(np.bool))
        for i_mc in range(2, len(data_mc_i_l_lst)):
            intersection = np.logical_and(intersection,
                                          data_mc_i_l_lst[i_mc].astype(np.bool))
            union = np.logical_or(union,
                                  data_mc_i_l_lst[i_mc].astype(np.bool))
        iou = np.sum(intersection) * 1. / np.sum(union)

        # compute coefficient of variation for all MC volume estimates for a given structure
        vol_mc_lst = [np.sum(data_mc_i_l_lst[i_mc]) for i_mc in range(len(data_mc_i_l_lst))]
        mu_mc = np.mean(vol_mc_lst)
        sigma_mc = np.std(vol_mc_lst)
        cv = sigma_mc / mu_mc

        # compute average voxel-wise uncertainty within the structure
        avgUnc = np.mean(data_uncVox[data_i_l != 0])
        # assign uncertainty value to the structure
        data_iou[data_i_l != 0] = iou
        data_cv[data_i_l != 0] = cv
        data_avgUnc[data_i_l != 0] = avgUnc

    # save nifti files
    fname_iou = fname_out.split('.nii.gz')[0] + '-iou.nii.gz'
    fname_cv = fname_out.split('.nii.gz')[0] + '-cv.nii.gz'
    fname_avgUnc = fname_out.split('.nii.gz')[0] + '-avgUnc.nii.gz'
    nib_iou = nib.Nifti1Image(data_iou, nib_hard.affine)
    nib_cv = nib.Nifti1Image(data_cv, nib_hard.affine)
    nib_avgUnc = nib.Nifti1Image(data_avgUnc, nib_hard.affine)
    nib.save(nib_iou, fname_iou)
    nib.save(nib_cv, fname_cv)
    nib.save(nib_avgUnc, fname_avgUnc)


def mixup(data, targets, alpha):
    """Compute the mixup data.
    Return mixed inputs and targets, lambda.
    """
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lambda_ = np.random.beta(alpha, alpha)
    lambda_ = max(lambda_, 1 - lambda_)  # ensure lambda_ >= 0.5
    lambda_tensor = torch.FloatTensor([lambda_])

    data = data * lambda_tensor + data2 * (1 - lambda_tensor)
    targets = targets * lambda_tensor + targets2 * (1 - lambda_tensor)

    return data, targets, lambda_tensor


def save_mixup_sample(x, y, fname):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(x, interpolation='nearest', aspect='auto', cmap='gray')

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(y, interpolation='nearest', aspect='auto', cmap='jet', vmin=0, vmax=1)

    plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


def segment_volume(folder_model, fname_image, fname_roi=None):
    """Segment an image.

    Segment an image (fname_image) using a pre-trained model (folder_model). If provided, a region of interest (fname_roi)
    is used to crop the image prior to segment it.

    Args:
        folder_model (string): foldername which contains
            (1) the model ('folder_model/folder_model.pt') to use
            (2) its configuration file ('folder_model/folder_model.json') used for the training,
                see https://github.com/neuropoly/ivado-medical-imaging/wiki/configuration-file
        fname_image (string): image filename (e.g. .nii.gz) to segment.
        fname_roi (string): Binary image filename (e.g. .nii.gz) defining a region of interest,
            e.g. spinal cord centerline, used to crop the image prior to segment it if provided.
            The segmentation is not performed on the slices that are empty in this image.
    Returns:
        nibabelObject: Object containing the soft segmentation.

    """
    # Define device
    device = torch.device("cpu")

    # Check if model folder exists
    # TODO: this check already exists in sct.deepseg.core.segment_nifti(). We should probably keep only one check in
    #  ivadomed, and do it in a separate, more specific file (e.g. models.py)
    if os.path.isdir(folder_model):
        prefix_model = os.path.basename(folder_model)
        # Check if model and model metadata exist
        fname_model = os.path.join(folder_model, prefix_model + '.pt')
        if not os.path.isfile(fname_model):
            print('Model file not found: {}'.format(fname_model))
            exit()
        fname_model_metadata = os.path.join(folder_model, prefix_model + '.json')
        if not os.path.isfile(fname_model_metadata):
            print('Model metadata file not found: {}'.format(fname_model_metadata))
            exit()
    else:
        print('Model folder not found: {}'.format(folder_model))
        exit()

    # Load model training config
    with open(fname_model_metadata, "r") as fhandle:
        context = json.load(fhandle)

    # If ROI is not provided then force center cropping
    if fname_roi is None and 'ROICrop' in context["transformation_validation"].keys():
        context["transformation_validation"] = dict((key, value) if key != 'ROICrop'
                                                    else ('CenterCrop', value)
                                                    for (key, value) in context["transformation_validation"].items())

    # Compose transforms
    do_transforms = imed_transforms.Compose(context['transformation_validation'], requires_undo=True)

    # Undo Transforms
    undo_transforms = imed_transforms.UndoCompose(do_transforms)

    # Force filter_empty_mask to False if fname_roi = None
    if fname_roi is None and 'filter_empty_mask' in context["slice_filter"]:
        context["slice_filter"]["filter_empty_mask"] = False

    fname_roi = [fname_roi] if fname_roi is not None else None

    # Load data
    filename_pairs = [([fname_image], None, fname_roi, [{}])]
    if not context['unet_3D']:  # TODO: rename this param 'model_name' or 'kernel_dim'
        ds = imed_loader.MRI2DSegmentationDataset(filename_pairs,
                                                  slice_axis=AXIS_DCT[context['slice_axis']],
                                                  cache=True,
                                                  transform=do_transforms,
                                                  slice_filter_fn=SliceFilter(**context["slice_filter"]))
    else:
        ds = imed_loader.MRI3DSubVolumeSegmentationDataset(filename_pairs,
                                                           transform=do_transforms,
                                                           length=context["length_3D"],
                                                           padding=context["padding_3D"])

    # If fname_roi provided, then remove slices without ROI
    if fname_roi is not None:
        ds = imed_loaded_utils.filter_roi(ds, nb_nonzero_thr=context["slice_filter_roi"])

    if not context['unet_3D']:
        print(f"\nLoaded {len(ds)} {context['slice_axis']} slices..")
    else:
        print(f"\nLoaded {len(ds)} {context['slice_axis']} volumes of shape {context['length_3D']}")

    # Data Loader
    data_loader = DataLoader(ds, batch_size=context["batch_size"],
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loaded_utils.imed_collate,
                             num_workers=0)

    # Load model
    model = torch.load(fname_model, map_location=device)

    # Inference time
    model.eval()

    # Loop across batches
    preds_list, slice_idx_list = [], []
    for i_batch, batch in enumerate(data_loader):
        with torch.no_grad():
            preds = model(batch['input'])

        # Reconstruct 3D object
        for i_slice in range(len(preds)):
            for i_label in range(preds[i_slice].shape[0]):
                batch["input_metadata"][i_slice][i_label]['data_type'] = 'gt'
            # undo transformations
            preds_i_undo, metadata_idx = undo_transforms(preds[i_slice],
                                                         batch["input_metadata"][i_slice],
                                                         data_type='gt')

            # Add new segmented slice to preds_list
            preds_list.append(np.array(preds_i_undo))
            # Store the slice index of preds_i_undo in the original 3D image
            slice_idx_list.append(int(batch['input_metadata'][i_slice][0]['slice_index']))

            # If last batch and last sample of this batch, then reconstruct 3D object
            if i_batch == len(data_loader) - 1 and i_slice == len(batch['gt']) - 1:
                pred_nib = pred_to_nib(data_lst=preds_list,
                                       fname_ref=fname_image,
                                       fname_out=None,
                                       z_lst=slice_idx_list,
                                       slice_axis=AXIS_DCT[context['slice_axis']],
                                       kernel_dim='3d' if context['unet_3D'] else '2d',
                                       debug=False,
                                       bin_thr=-1)

    return pred_nib


def cuda(input_var, non_blocking=False):
    """
    This function sends input_var to GPU.
    :param input_var: either a tensor or a list of tensors
    :param non_blocking
    :return: same as input_var
    """

    if isinstance(input_var, list):
        return [t.cuda(non_blocking=non_blocking) for t in input_var]
    else:
        return input_var.cuda(non_blocking=non_blocking)


class HookBasedFeatureExtractor(nn.Module):
    """
    This function extracts feature maps from given layer.
    https://github.com/ozan-oktay/Attention-Gated-Networks/tree/a96edb72622274f6705097d70cfaa7f2bf818a5a
    """

    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


def reorient_image(arr, slice_axis, nib_ref, nib_ref_canonical):
    # Orient image in RAS according to slice axis
    arr_ras = imed_loaded_utils.orient_img_ras(arr, slice_axis)

    # https://gitship.com/neuroscience/nibabel/blob/master/nibabel/orientations.py
    ref_orientation = nib.orientations.io_orientation(nib_ref.affine)
    ras_orientation = nib.orientations.io_orientation(nib_ref_canonical.affine)
    # Return the orientation that transforms from ras to ref_orientation
    trans_orient = nib.orientations.ornt_transform(ras_orientation, ref_orientation)
    # apply transformation
    return nib.orientations.apply_orientation(arr_ras, trans_orient)


def save_feature_map(batch, layer_name, context, model, test_input, slice_axis):
    if not os.path.exists(os.path.join(context["log_directory"], layer_name)):
        os.mkdir(os.path.join(context["log_directory"], layer_name))

    # Save for subject in batch
    for i in range(batch['input'].size(0)):
        inp_fmap, out_fmap = \
            HookBasedFeatureExtractor(model, layer_name, False).forward(Variable(test_input[i][None,]))

        # Display the input image and Down_sample the input image
        orig_input_img = test_input[i][None,].permute(3, 4, 2, 0, 1).cpu().numpy()
        upsampled_attention = F.interpolate(out_fmap[1],
                                            size=test_input[i][None,].size()[2:],
                                            mode='trilinear',
                                            align_corners=True).data.permute(3, 4, 2, 0, 1).cpu().numpy()

        path = batch["input_metadata"][0][i]["input_filenames"]

        basename = path.split('/')[-1]
        save_directory = os.path.join(context['log_directory'], layer_name, basename)

        # Write the attentions to a nifti image
        nib_ref = nib.load(path)
        nib_ref_can = nib.as_closest_canonical(nib_ref)
        oriented_image = reorient_image(orig_input_img[:, :, :, 0, 0], slice_axis, nib_ref, nib_ref_can)

        nib_pred = nib.Nifti1Image(oriented_image, nib_ref.affine)
        nib.save(nib_pred, save_directory)

        basename = basename.split(".")[0] + "_att.nii.gz"
        save_directory = os.path.join(context['log_directory'], layer_name, basename)
        attention_map = reorient_image(upsampled_attention[:, :, :, 0, ], slice_axis, nib_ref, nib_ref_can)
        nib_pred = nib.Nifti1Image(attention_map, nib_ref.affine)

        nib.save(nib_pred, save_directory)


def save_color_labels(gt_data, binarize, gt_filename, output_filename, slice_axis):
    rdict = {}
    n_class, h, w, d = gt_data.shape
    labels = range(n_class)
    # Generate color labels
    multi_labeled_pred = np.zeros((h, w, d, 3))
    if binarize:
        rdict['gt'] = imed_postpro.threshold_predictions(gt_data)

    # Keep always the same color labels
    np.random.seed(6)

    for label in labels:
        r, g, b = np.random.randint(0, 256, size=3)
        multi_labeled_pred[..., 0] += r * gt_data[label,]
        multi_labeled_pred[..., 1] += g * gt_data[label,]
        multi_labeled_pred[..., 2] += b * gt_data[label,]

    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    multi_labeled_pred = multi_labeled_pred.copy().astype('u1').view(dtype=rgb_dtype).reshape((h, w, d))

    pred_to_nib([multi_labeled_pred], [], gt_filename,
                output_filename, slice_axis=slice_axis, kernel_dim='3d', bin_thr=-1, discard_noise=False)

    return multi_labeled_pred


def convert_labels_to_RGB(grid_img):
    # Keep always the same color labels
    batch_size, n_class, h, w = grid_img.shape
    rgb_img = torch.zeros((batch_size, 3, h, w))

    # Keep always the same color labels
    np.random.seed(6)
    for i in range(n_class):
        r, g, b = np.random.randint(0, 256, size=3)
        rgb_img[:, i, ] = r * grid_img[:, i, ]
        rgb_img[:, i, ] = g * grid_img[:, i, ]
        rgb_img[:, i, ] = b * grid_img[:, i, ]

    return rgb_img


def save_tensorboard_img(writer, epoch, dataset_type, input_samples, gt_samples, preds, unet_3D):
    if unet_3D:
        # Take all images stacked on depth dimension
        num_2d_img = input_samples.shape[-1]
    else:
        num_2d_img = 1
    if isinstance(input_samples, list):
        input_samples_copy = input_samples.copy()
    else:
        input_samples_copy = input_samples.clone()
    preds_copy = preds.clone()
    gt_samples_copy = gt_samples.clone()
    for idx in range(num_2d_img):
        if unet_3D:
            input_samples = input_samples_copy[..., idx]
            preds = preds_copy[..., idx]
            gt_samples = gt_samples_copy[..., idx]
            # Only display images with labels
            if gt_samples.sum() == 0:
                continue

        # take only one modality for grid
        if not isinstance(input_samples, list) and input_samples.shape[1] > 1:
            tensor = input_samples[:, 0, ][:, None, ]
            input_samples = torch.cat((tensor, tensor, tensor), 1)
        elif isinstance(input_samples, list):
            input_samples = input_samples[0]

        grid_img = vutils.make_grid(input_samples,
                                    normalize=True,
                                    scale_each=True)
        writer.add_image(dataset_type + '/Input', grid_img, epoch)

        grid_img = vutils.make_grid(convert_labels_to_RGB(preds),
                                    normalize=True,
                                    scale_each=True)

        writer.add_image(dataset_type + '/Predictions', grid_img, epoch)

        grid_img = vutils.make_grid(convert_labels_to_RGB(gt_samples),
                                    normalize=True,
                                    scale_each=True)

        writer.add_image(dataset_type + '/Ground Truth', grid_img, epoch)


class SliceFilter(object):
    def __init__(self, filter_empty_mask=True,
                 filter_empty_input=True):
        self.filter_empty_mask = filter_empty_mask
        self.filter_empty_input = filter_empty_input

    def __call__(self, sample):
        input_data, gt_data = sample['input'], sample['gt']

        if self.filter_empty_mask:
            if not np.any(gt_data):
                return False

        if self.filter_empty_input:
            if not np.all([np.any(img) for img in input_data]):
                return False

        return True


def unstack_tensors(sample):
    list_tensor = []
    for i in range(sample.shape[1]):
        list_tensor.append(sample[:, i, ].unsqueeze(1))
    return list_tensor

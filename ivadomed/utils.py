import torch
import numpy as np
import nibabel as nib
from PIL import Image
from skimage.measure import label
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

from medicaltorch import filters as mt_filters
from medicaltorch import metrics as mt_metrics


class IvadoMetricManager(mt_metrics.MetricManager):
    def __init__(self, metric_fns):
        super().__init__(metric_fns)

        self.result_dict = defaultdict(list)

    def __call__(self, prediction, ground_truth):
        self.num_samples += len(prediction)
        for metric_fn in self.metric_fns:
            for p, gt in zip(prediction, ground_truth):
                res = metric_fn(p, gt)
                dict_key = metric_fn.__name__
                self.result_dict[dict_key].append(res)

    def get_results(self):
        res_dict = {}
        for key, val in self.result_dict.items():
            if np.all(np.isnan(val)):  # if all values are np.nan
                res_dict[key] = None
            else:
                res_dict[key] = np.nanmean(val)
        return res_dict


class Evaluation3DMetrics(object):

    def __init__(self, fname_pred, fname_gt, params=None):
        self.fname_pred = fname_pred
        self.fname_gt = fname_gt

        if not params is None:
            pass

        self.data_pred = self.get_data(self.fname_pred)
        self.data_gt = self.get_data(self.fname_gt)

        self.px, self.py, self.pz = self.get_pixdim(self.fname_pred)

        # Note: Some papers suggested to remove all lesion with less than 3 voxels: Todo?

        # 18-connected components
        self.data_pred_label, self.n_pred = label(self.data_pred,
                                                    connectivity=3,
                                                    return_num=True)
        self.data_gt_label, self.n_gt = label(self.data_gt,
                                                connectivity=3,
                                                return_num=True)

    def get_data(self, fname):
        nib_im = nib.load(fname)
        return nib_im.get_data()

    def get_pixdim(self, fname):
        nib_im = nib.load(fname)
        px, py, pz = nib_im.header['pixdim'][1:4]
        return px, py, pz

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

        rvd = (vol_gt-vol_pred)*100.
        rvd /= vol_gt
        return rvd

    def get_avd(self):
        """Absolute volume difference."""
        return abs(self.get_rvd())

    def _get_ltp_lfn(self, overlap_vox=3):
        """Number of true positive and false negative lesion.

            Note1: if two lesion_pred overlap with the current lesion_gt,
                then only one detection is counted.
            Note2: the tolerance of overlap that define a detection is
                expressed in voxels. We could change it to "relative volume".
        """
        ltp, lfn = 0, 0
        for idx in range(1, self.n_gt+1):
            data_gt_idx = (self.data_gt_label == idx).astype(np.int)
            overlap = (data_gt_idx * self.data_pred).astype(np.int)

            if np.count_nonzero(overlap) >= overlap_vox:
                ltp += 1

            else:
                lfn += 1

        return ltp, lfn

    def _get_lfp(self, overlap_vox=3):
        """Number of false positive lesion."""
        lfp = 0
        for idx in range(1, self.n_pred+1):
            data_pred_idx = (self.data_pred_label == idx).astype(np.int)
            overlap = (data_pred_idx * self.data_gt).astype(np.int)

            if np.count_nonzero(overlap) < overlap_vox:
                lfp += 1

        return lfp

    def get_ltpr(self):
        """Lesion True Positive Rate / Recall / Sensitivity.

            Note: computed only if self.n_gt >= 1.
        """
        ltp, lfn = self._get_ltp_lfn()

        denom = ltp + lfn
        if denom == 0 or self.n_gt == 0:
            return np.nan

        return ltp * 100. / denom

    def get_lfdr(self):
        """Lesion False Detection Rate / 1 - Precision.

            Note: computed only if self.n_gt >= 1.
        """
        ltp, _ = self._get_ltp_lfn()
        lfp = self._get_lfp()

        denom = ltp + lfp
        if denom == 0 or self.n_gt == 0:
            return np.nan

        return lfp * 100. / denom

    def get_all_metrics(self):
        dct = {}
        dct['vol_pred'] = self.get_vol(self.data_pred)
        dct['vol_gt'] = self.get_vol(self.data_gt)
        dct['rvd'], dct['avd'] = self.get_rvd(), self.get_avd()
        dct['dice'] = dice_score(self.data_gt, self.data_pred) * 100.
        dct['recall'] = mt_metrics.recall_score(self.data_pred, self.data_gt, err_value=np.nan)
        dct['precision'] = mt_metrics.precision_score(self.data_pred, self.data_gt, err_value=np.nan)
        dct['specificity'] = mt_metrics.specificity_score(self.data_pred, self.data_gt, err_value=np.nan)
        dct['n_pred'], dct['n_gt'] = self.n_pred, self.n_gt
        dct['ltpr'] = self.get_ltpr()
        dct['lfdr'] = self.get_lfdr()

        return dct


def save_nii(data_lst, z_lst, fname_ref, fname_out, slice_axis, debug=False):
    """Save the prediction as nii.
        1. Reconstruct a 3D volume out of the slice-wise predictions.
        2. Re-orient the 3D array accordingly to the ground-truth orientation.

    Inputs:
        data_lst: list of the slice-wise predictions.
        z_lst: list of the slice indexes where the inference has been performed.
              The remaining slices will be filled with zeros.
        fname_ref: ground-truth fname
        fname_out: output fname
        slice_axis: orientation used to extract slices (i.e. axial, sagittal, coronal)
        debug:

    Return:

    """
    nib_ref = nib.load(fname_ref)
    nib_ref_can = nib.as_closest_canonical(nib_ref)

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

    # create data
    arr = np.stack(tmp_lst, axis=0)
    if slice_axis == 2:
        arr = np.swapaxes(arr, 1, 2)
    # move axis according to slice_axis to RAS orientation
    arr_ras = np.swapaxes(arr, 0, slice_axis)

    # https://gitship.com/neuroscience/nibabel/blob/master/nibabel/orientations.py
    ref_orientation = nib.orientations.io_orientation(nib_ref.affine)
    ras_orientation = nib.orientations.io_orientation(nib_ref_can.affine)
    # Return the orientation that transforms from ras to ref_orientation
    trans_orient = nib.orientations.ornt_transform(ras_orientation, ref_orientation)
    # apply transformation
    arr_pred_ref_space = nib.orientations.apply_orientation(arr_ras, trans_orient)

    # create nii
    nib_pred = nib.Nifti1Image(arr_pred_ref_space, nib_ref.affine)

    # save
    nib.save(nib_pred, fname_out)


def dice_score(im1, im2):
    """
    Computes the Dice coefficient between im1 and im2.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return np.nan

    intersection = np.logical_and(im1, im2)
    return (2. * intersection.sum())/ im_sum


def mixup(data, targets, alpha):
    """Compute the mixup data.
    Return mixed inputs and targets, lambda.
    """
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lambda_ = np.random.beta(alpha, alpha)
    lambda_ = max(lambda_, 1 - lambda_) # ensure lambda_ >= 0.5
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


def threshold_predictions(predictions, thr=0.5):
    """This function will threshold predictions.
    :param predictions: input data (predictions)
    :param thr: threshold to use, default to 0.5
    :return: thresholded input
    """
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds



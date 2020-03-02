import os
import torch
from tqdm import tqdm
import numpy as np
import nibabel as nib
from PIL import Image
from scipy.ndimage import label, generate_binary_structure
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

from medicaltorch import filters as mt_filters
from medicaltorch import metrics as mt_metrics

# labels of paint_objects method
TP_COLOUR=1
FP_COLOUR=2
FN_COLOUR=3

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

    def __init__(self, fname_pred, fname_gt, params={}):
        self.fname_pred = fname_pred
        self.fname_gt = fname_gt

        self.data_pred = self.get_data(self.fname_pred)
        self.data_gt = self.get_data(self.fname_gt)

        self.px, self.py, self.pz = self.get_pixdim(self.fname_pred)

        self.bin_struct = generate_binary_structure(3, 2)  # 18-connectivity

        # Remove small objects
        if "removeSmall" in params:
            size_min = params['removeSmall']['thr']
            if params['removeSmall']['unit'] == 'vox':
                self.size_min = size_min
            elif params['removeSmall']['unit'] == 'mm3':
                self.size_min = np.round(size_min / (self.px * self.py * self.pz))
            else:
                raise NotImplmentedError

            self.data_pred = self.remove_small_objects(data=self.data_pred)
            self.data_gt = self.remove_small_objects(data=self.data_gt)
        else:
            self.size_min = 0

        if "targetSize" in params:
            self.size_rng_lst, self.size_suffix_lst =
                                    self._get_size_ranges(thr_lst=params["targetSize"]["thr"],
                                                                unit=params["targetSize"]["unit"])
            self.data_per_size = self.label_per_size(self.data_gt, rng_lst)

        # 18-connected components
        self.data_pred_label, self.n_pred = label(self.data_pred,
                                                    structure=self.bin_struct)
        self.data_gt_label, self.n_gt = label(self.data_gt,
                                                structure=self.bin_struct)
        print('After: GT {}, Pred {}'.format(self.n_gt, self.n_pred))
        # painted data, object wise
        self.fname_paint = fname_pred.split('.nii.gz')[0] + '_painted.nii.gz'
        self.data_painted = np.copy(self.data_pred)

        if "targetSize" in params:
            self.targetSize = params["targetSize"]

        else:
            self.targetSize = None

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

    def get_data(self, fname):
        nib_im = nib.load(fname)
        return nib_im.get_data()

    def get_pixdim(self, fname):
        nib_im = nib.load(fname)
        px, py, pz = nib_im.header['pixdim'][1:4]
        return px, py, pz

    def remove_small_objects(self, data):
        data_label, n = label(data,
                                structure=self.bin_struct)

        for idx in range(1, n+1):
            data_idx = (data_label == idx).astype(np.int)
            n_nonzero = np.count_nonzero(data_idx)

            if n_nonzero < self.size_min:
                data[data_label == idx] = 0

        return data

    def _get_size_ranges(thr_lst, unit):
        assert unit in ['vox', 'mm3']

        rng_lst, suffix_lst = [], []
        for i, thr in enumerate(thr_lst):
            if i == 0:
                thr_low = self.size_min
            else:
                thr_low = thr_lst[i-1] + 1

            thr_high = thr

            if unit == 'mm3':
                thr_low = np.round(thr_low / (self.px * self.py * self.pz))
                thr_high = np.round(thr_high / (self.px * self.py * self.pz))

            rng_lst.append([thr_low, thr_high])

            suffix_lst.append('_'+str(thr_low)+'-'+str(thr_high)+unit)

        # last subgroup
        thr_low = thr_lst[i] + 1
        if unit == 'mm3':
            thr_low = np.round(thr_low / (self.px * self.py * self.pz))
        thr_high = np.inf
        rng_lst.append([thr_low, thr_high])
        suffix_lst.append('_'+str(thr_low)+'-INF'+unit)

        return rng_lst, suffix_lst

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

    def _get_ltp_lfn(self):
        """Number of true positive and false negative lesion.

            Note1: if two lesion_pred overlap with the current lesion_gt,
                then only one detection is counted.
        """
        ltp, lfn = 0, 0
        for idx in range(1, self.n_gt+1):
            data_gt_idx = (self.data_gt_label == idx).astype(np.int)
            overlap = (data_gt_idx * self.data_pred).astype(np.int)

            if self.overlap_vox is None:
                overlap_vox = np.round(np.count_nonzero(data_gt_idx) * self.overlap_percent / 100.)
            else:
                overlap_vox = self.overlap_vox

            if np.count_nonzero(overlap) >= overlap_vox:
                ltp += 1

            else:
                lfn += 1
                self.data_painted[self.data_gt_label == idx] = FN_COLOUR

        return ltp, lfn

    def _get_lfp(self):
        """Number of false positive lesion."""
        lfp = 0
        for idx in range(1, self.n_pred+1):
            data_pred_idx = (self.data_pred_label == idx).astype(np.int)
            overlap = (data_pred_idx * self.data_gt).astype(np.int)

            if self.overlap_vox is None:
                label_gt = np.max(data_pred_idx * self.data_label_gt)
                data_gt_idx = (self.data_gt_label == label_gt).astype(np.int)
                overlap_thr = np.round(np.count_nonzero(data_gt_idx) * self.overlap_percent / 100.)
            else:
                overlap_thr = self.overlap_vox

            if np.count_nonzero(overlap) < overlap_thr:
                lfp += 1
                self.data_painted[self.data_pred_label == idx] = FP_COLOUR
            else:
                self.data_painted[self.data_pred_label == idx] = TP_COLOUR

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

    def run_eval(self):
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

        # save painted file
        nib_painted = nib.Nifti1Image(self.data_painted, nib.load(self.fname_pred).affine)
        nib.save(nib_painted, self.fname_paint)

        return dct


def save_nii(data_lst, z_lst, fname_ref, fname_out, slice_axis, debug=False, unet_3D=False, binarize=True):
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

    if not unet_3D:
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

    else:
        arr = data_lst

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
    if binarize:
        arr_pred_ref_space = threshold_predictions(arr_pred_ref_space)

    # create nii
    nib_pred = nib.Nifti1Image(arr_pred_ref_space, nib_ref.affine)


    # save
    nib.save(nib_pred, fname_out)


def run_uncertainty(ifolder):
    # list subj_acq prefixes
    subj_acq_lst = [f.split('_pred')[0] for f in os.listdir(ifolder)
                    if f.endswith('.nii.gz') and '_pred' in f]
    # remove duplicates
    subj_acq_lst = list(set(subj_acq_lst))
    # keep only the images where unc has not been computed yet
    subj_acq_lst = [f for f in subj_acq_lst if not os.path.isfile(
        os.path.join(ifolder, f+'_unc-cv.nii.gz'))]

    # loop across subj_acq
    for subj_acq in tqdm(subj_acq_lst, desc="Uncertainty Computation"):
        # hard segmentation from MC samples
        fname_pred = os.path.join(ifolder, subj_acq+'_pred.nii.gz')
        # fname for soft segmentation from MC simulations
        fname_soft = os.path.join(ifolder, subj_acq+'_soft.nii.gz')
        # find Monte Carlo simulations
        fname_pred_lst = [os.path.join(ifolder, f)
                          for f in os.listdir(ifolder) if subj_acq+'_pred_' in f]

        # if final segmentation from Monte Carlo simulations has not been generated yet
        if not os.path.isfile(fname_pred) or not os.path.isfile(fname_soft):
            # threshold used for the hard segmentation
            thr = 1. / len(fname_pred_lst)  # 1 for all voxels where at least on MC sample predicted 1
            # average then argmax
            combine_predictions(fname_pred_lst, fname_pred, fname_soft, thr=thr)

        fname_unc_vox = os.path.join(ifolder, subj_acq+'_unc-vox.nii.gz')
        if not os.path.isfile(fname_unc_vox):
            # compute voxel-wise uncertainty map
            voxelWise_uncertainty(fname_pred_lst, fname_unc_vox)

        fname_unc_struct = os.path.join(ifolder, subj_acq+'_unc.nii.gz')
        if not os.path.isfile(os.path.join(ifolder, subj_acq+'_unc-cv.nii.gz')):
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
    data_hard = threshold_predictions(data_prob, thr=thr).astype(np.uint8)
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
    unc = np.repeat(np.expand_dims(np.array(data_lst), -1), 2, -1) # n_it, x, y, z, 2
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
    for i_l in range(1, n_l+1):
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
    fname_iou = fname_out.split('.nii.gz')[0]+'-iou.nii.gz'
    fname_cv = fname_out.split('.nii.gz')[0]+'-cv.nii.gz'
    fname_avgUnc = fname_out.split('.nii.gz')[0]+'-avgUnc.nii.gz'
    nib_iou = nib.Nifti1Image(data_iou, nib_hard.affine)
    nib_cv = nib.Nifti1Image(data_cv, nib_hard.affine)
    nib_avgUnc = nib.Nifti1Image(data_avgUnc, nib_hard.affine)
    nib.save(nib_iou, fname_iou)
    nib.save(nib_cv, fname_cv)
    nib.save(nib_avgUnc, fname_avgUnc)


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
    return (2. * intersection.sum()) / im_sum


def hausdorff_score(prediction, groundtruth):
    if len(prediction.shape) == 3:
        mean_hansdorff = 0
        for idx in range(prediction.shape[1]):
            pred = prediction[:, idx, :]
            gt = groundtruth[:, idx, :]
            mean_hansdorff += mt_metrics.hausdorff_score(pred, gt)
        mean_hansdorff = mean_hansdorff / prediction.shape[1]
        return mean_hansdorff

    return mt_metrics.hausdorff_score(prediction, groundtruth)


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


def cuda(input_var):
    """
    This function sends input_var to GPU.
    :param input_var: either a tensor or a list of tensors
    :return: same as input_var
    """

    if isinstance(input_var, list):
        return [t.cuda() for t in input_var]
    else:
        return input_var.cuda()

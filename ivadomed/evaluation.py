import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label, generate_binary_structure
from tqdm import tqdm

from ivadomed import metrics as imed_metrics
from ivadomed import postprocessing as imed_postpro

# labels of paint_objects method
TP_COLOUR = 1
FP_COLOUR = 2
FN_COLOUR = 3


def evaluate(bids_path, log_directory, target_suffix, eval_params):
    """Evaluate predictions from inference step.

    Args:
        bids_path (str): Folder where raw data is stored.
        log_directory (str): Folder where the output folder "results_eval" is be created.
        target_suffix (list): List of suffixes that indicates the target mask(s).
        eval_params (dict): Evaluation parameters.

    Returns:
        pd.Dataframe: results for each image.
    """
    path_preds = os.path.join(log_directory, 'pred_masks')
    print('\nRun Evaluation on {}\n'.format(path_preds))

    # OUTPUT RESULT FOLDER
    path_results = os.path.join(log_directory, 'results_eval')
    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    # INIT DATA FRAME
    df_results = pd.DataFrame()

    # LIST PREDS
    subj_acq_lst = [f.split('_pred')[0] for f in os.listdir(path_preds) if f.endswith('_pred.nii.gz')]

    # LOOP ACROSS PREDS
    for subj_acq in tqdm(subj_acq_lst, desc="Evaluation"):
        # Fnames of pred and ground-truth
        subj, acq = subj_acq.split('_')[0], '_'.join(subj_acq.split('_')[1:])
        fname_pred = os.path.join(path_preds, subj_acq + '_pred.nii.gz')
        fname_gt = [os.path.join(bids_path, 'derivatives', 'labels', subj, 'anat', subj_acq + suffix + '.nii.gz')
                    for suffix in target_suffix]
        # Uncertainty
        data_uncertainty = None

        # 3D evaluation
        nib_pred = nib.load(fname_pred)
        data_pred = nib_pred.get_fdata()

        h, w, d = data_pred.shape[:3]
        n_classes = len(fname_gt)
        data_gt = np.zeros((h, w, d, n_classes))
        for idx, file in enumerate(fname_gt):
            if os.path.exists(file):
                data_gt[..., idx] = nib.load(file).get_fdata()
            else:
                data_gt[..., idx] = np.zeros((h, w, d), dtype='u1')
        eval = Evaluation3DMetrics(data_pred=data_pred,
                                   data_gt=data_gt,
                                   dim_lst=nib_pred.header['pixdim'][1:4],
                                   params=eval_params)
        results_pred, data_painted = eval.run_eval()

        # SAVE PAINTED DATA, TP FP FN
        fname_paint = fname_pred.split('.nii.gz')[0] + '_painted.nii.gz'
        nib_painted = nib.Nifti1Image(data_painted, nib_pred.affine)
        nib.save(nib_painted, fname_paint)

        # SAVE RESULTS FOR THIS PRED
        results_pred['image_id'] = subj_acq
        df_results = df_results.append(results_pred, ignore_index=True)

    df_results = df_results.set_index('image_id')
    df_results.to_csv(os.path.join(path_results, 'evaluation_3Dmetrics.csv'))

    print(df_results.head(5))
    return df_results


class Evaluation3DMetrics(object):
    """Computes 3D evaluation metrics.

    Args:
        data_pred (ndarray): Network prediction mask.
        data_gt (ndarray): Ground-truth mask.
        dim_lst (list): Resolution (mm) along each dimension.
        params (dict): Evaluation parameters.

    Attributes:
        data_pred (ndarray): Network prediction mask.
        data_gt (ndarray): Ground-truth mask.
        n_classes (int): Number of classes.
        px (float): Resolution (mm) along the first axis.
        py (float): Resolution (mm) along the second axis.
        pz (float): Resolution (mm) along the third axis.
        bin_struct (ndarray): Binary structure.
        size_min (int): Minimum size of objects. Objects that are smaller than this limit can be removed if
            "removeSmall" is in params.
        overlap_vox (int): A prediction and ground-truth are considered as overlapping if they overlap for at least this
            amount of voxels.
        overlap_ratio (float): A prediction and ground-truth are considered as overlapping if they overlap for at least
            this portion of their volumes.
        data_pred_label (ndarray): Network prediction mask that is labeled, ie each object is filled with a different
            value.
        data_gt_label (ndarray): Ground-truth mask that is labeled, ie each object is filled with a different
            value.
        n_pred (int): Number of objects in the network prediction mask.
        n_gt (int): Number of objects in the ground-truth mask.
        data_painted (ndarray): Mask where each predicted object is labeled depending on whether it is a TP or FP.
    """

    def __init__(self, data_pred, data_gt, dim_lst, params=None):
        if params is None:
            params = {}

        self.data_pred = data_pred
        if len(self.data_pred.shape) == 3:
            self.data_pred = np.expand_dims(self.data_pred, -1)

        self.data_gt = data_gt
        if len(self.data_gt.shape) == 3:
            self.data_gt = np.expand_dims(self.data_gt, -1)

        h, w, d, self.n_classes = self.data_gt.shape
        self.px, self.py, self.pz = dim_lst

        self.bin_struct = generate_binary_structure(3, 2)  # 18-connectivity
        self.postprocessing_dict = {}
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
            elif params["overlap"]["unit"] == 'ratio':  # The ratio of the GT object
                self.overlap_ratio = params["overlap"]["thr"]
                self.overlap_vox = None
        else:
            self.overlap_vox = 3

    def _get_size_ranges(self, thr_lst, unit):
        """Get size ranges of objects in image.

        Args:
            thr_lst (list): Bins ranging each size category.
            unit (str): Choice between 'vox' for voxel of 'mm3'.

        Returns:
            list, list: range list, suffix related to range
        """
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
        """Get data with labels corresponding to label size.

        Args:
            data (ndarray): Input data.

        Returns:
            ndarray
        """
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
        """Get volume."""
        vol = np.sum(data)
        vol *= self.px * self.py * self.pz
        return vol

    def get_rvd(self):
        """Relative volume difference.

        The volume is here defined by the physical volume, in mm3, of the non-zero voxels of a given mask.
        Relative volume difference equals the difference between the ground-truth and prediction volumes, divided by the
        ground-truth volume.
        Optimal value is zero. Negative value indicates over-segmentation, while positive value indicates
        under-segmentation.
        """
        vol_gt = self.get_vol(self.data_gt)
        vol_pred = self.get_vol(self.data_pred)

        if vol_gt == 0.0:
            return np.nan

        rvd = (vol_gt - vol_pred)
        rvd /= vol_gt

        return rvd

    def get_avd(self):
        """Absolute volume difference.

        The volume is here defined by the physical volume, in mm3, of the non-zero voxels of a given mask.
        Absolute volume difference equals the absolute value of the Relative Volume Difference.
        Optimal value is zero.
        """
        return abs(self.get_rvd())

    def _get_ltp_lfn(self, label_size, class_idx=0):
        """Number of true positive and false negative lesion.

        Args:
            label_size (int): Size of label.
            class_idx (int): Label index. If monolabel 0, else ranges from 0 to number of output channels - 1.

        Note1: if two lesion_pred overlap with the current lesion_gt,
            then only one detection is counted.
        """
        ltp, lfn, n_obj = 0, 0, 0

        for idx in range(1, self.n_gt[class_idx] + 1):
            data_gt_idx = (self.data_gt_label[..., class_idx] == idx).astype(np.int)
            overlap = (data_gt_idx * self.data_pred).astype(np.int)

            # if label_size is None, then we look at all object sizes
            # we check if the currrent object belongs to the current size range
            if label_size is None or \
                    np.max(self.data_gt_per_size[..., class_idx][np.nonzero(data_gt_idx)]) == label_size:

                if self.overlap_vox is None:
                    overlap_vox = np.round(np.count_nonzero(data_gt_idx) * self.overlap_ratio)
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
        """Number of false positive lesion.

        Args:
            label_size (int): Size of label.
            class_idx (int): Label index. If monolabel 0, else ranges from 0 to number of output channels - 1.
        """
        lfp = 0
        for idx in range(1, self.n_pred[class_idx] + 1):
            data_pred_idx = (self.data_pred_label[..., class_idx] == idx).astype(np.int)
            overlap = (data_pred_idx * self.data_gt).astype(np.int)

            label_gt = np.max(data_pred_idx * self.data_gt_label[..., class_idx])
            data_gt_idx = (self.data_gt_label[..., class_idx] == label_gt).astype(np.int)
            # if label_size is None, then we look at all object sizes
            # we check if the current object belongs to the current size range

            if label_size is None or \
                    np.max(self.data_pred_per_size[..., class_idx][np.nonzero(data_gt_idx)]) == label_size:

                if self.overlap_vox is None:
                    overlap_thr = np.round(np.count_nonzero(data_gt_idx) * self.overlap_ratio)
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

        Args:
            label_size (int): Size of label.
            class_idx (int): Label index. If monolabel 0, else ranges from 0 to number of output channels - 1.

        Note: computed only if n_obj >= 1.
        """
        ltp, lfn, n_obj = self._get_ltp_lfn(label_size, class_idx)

        denom = ltp + lfn
        if denom == 0 or n_obj == 0:
            return np.nan, n_obj

        return ltp / denom, n_obj

    def get_lfdr(self, label_size=None, class_idx=0):
        """Lesion False Detection Rate / 1 - Precision.

        Args:
            label_size (int): Size of label.
            class_idx (int): Label index. If monolabel 0, else ranges from 0 to number of output channels - 1.

        Note: computed only if n_obj >= 1.
        """
        ltp, _, n_obj = self._get_ltp_lfn(label_size, class_idx)
        lfp = self._get_lfp(label_size, class_idx)

        denom = ltp + lfp
        if denom == 0 or n_obj == 0:
            return np.nan

        return lfp / denom

    def run_eval(self):
        """Stores evaluation results in dictionary

        Returns:
            dict, ndarray: dictionary containing evaluation results, data with each object painted a different color
        """
        dct = {}
        data_gt = self.data_gt.copy()
        data_pred = self.data_pred.copy()
        for n in range(self.n_classes):
            self.data_pred = data_pred[..., n]
            self.data_gt = data_gt[..., n]
            dct['vol_pred_class' + str(n)] = self.get_vol(self.data_pred)
            dct['vol_gt_class' + str(n)] = self.get_vol(self.data_gt)
            dct['rvd_class' + str(n)], dct['avd_class' + str(n)] = self.get_rvd(), self.get_avd()
            dct['dice_class' + str(n)] = imed_metrics.dice_score(self.data_gt, self.data_pred)
            dct['recall_class' + str(n)] = imed_metrics.recall_score(self.data_pred, self.data_gt, err_value=np.nan)
            dct['precision_class' + str(n)] = imed_metrics.precision_score(self.data_pred, self.data_gt,
                                                                           err_value=np.nan)
            dct['specificity_class' + str(n)] = imed_metrics.specificity_score(self.data_pred, self.data_gt,
                                                                               err_value=np.nan)
            dct['n_pred_class' + str(n)], dct['n_gt_class' + str(n)] = self.n_pred[n], self.n_gt[n]
            dct['ltpr_class' + str(n)], _ = self.get_ltpr(class_idx=n)
            dct['lfdr_class' + str(n)] = self.get_lfdr(class_idx=n)
            dct['mse_class' + str(n)] = imed_metrics.mse(self.data_gt, self.data_pred)

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

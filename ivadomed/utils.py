import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes, binary_closing

from medicaltorch import filters as mt_filters
from medicaltorch import transforms as mt_transforms
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


class DilateGT(mt_transforms.MTTransform):
    """Randomly dilate a tensor ground-truth.
    :param nb_dilation_it: Number of dilation iterations1.
    """
    def __init__(self, nb_dilation_it):
        self.n_dil_it = nb_dilation_it

    @staticmethod
    def dilate_mask(arr, label_values):
        arr_bin, arr_soft = arr.astype(np.int), arr.astype(np.float)

        for lb in label_values:
            # binary dilation with 1 iteration
            arr_dilated = binary_dilation(arr_bin, iterations=1)

            # isolate new voxels, i.e. the ones from the dilation
            new_voxels = np.logical_xor(arr_dilated, arr_bin).astype(np.int)

            # assign a soft value (]0, 1[) to the new voxels
            soft_new_voxels = lb * new_voxels

            # add the new voxels to the input mask
            arr_soft += soft_new_voxels
            arr_bin = (arr_soft > 0).astype(np.int)

        return arr_soft, arr_bin

    @staticmethod
    def random_holes(arr_in, arr_soft, arr_bin):
        for idx in range(arr_in.shape[0]):
            # coordinates of the new voxels, i.e. the ones from the dilation
            new_voxels_xx, new_voxels_yy = np.where(np.logical_xor(arr_bin[idx, 0], arr_in[idx, 0]))
            nb_new_voxels = new_voxels_xx.shape[0]

            # ratio of voxels added to the input mask from the dilated mask
            new_voxel_ratio = random.random()
            # randomly select new voxel indexes to remove
            idx_to_remove = random.sample(range(nb_new_voxels),
                                            int(round(nb_new_voxels * (1 - new_voxel_ratio))))
            # set to zero the here-above randomly selected new voxels
            arr_soft[idx, 0, new_voxels_xx[idx_to_remove], new_voxels_yy[idx_to_remove]] = 0.0

       arr_bin = (arr_soft > 0).astype(np.int)

       return arr_soft, arr_bin

    @staticmethod
    def post_processing(arr_in, arr_soft, arr_bin, arr_dil):
        # remove new object that are not connected to the input mask
        arr_labeled, lb_nb = label(arr_bin)
        connected_to_in = arr_labeled * arr_in
        for lb in range(1, lb_nb+1):
            if np.sum(connected_to_in == lb) == 0:
                arr_soft[arr_labeled == lb] = 0

        # fill binary holes
        arr_bin_filled = binary_fill_holes((arr_soft > 0).astype(np.int))
        # binary closing
        arr_bin_closed = binary_closing(arr_bin_filled.astype(np.int))
        # recover the soft-value assigned to the filled-holes
        arr_soft_out = arr_bin_closed * arr_dil

        return arr_soft_out

    def __call__(self, sample):
        gt_data = sample['gt']
        gt_data_np = gt_data.numpy()

        # values of the voxels added to the input mask
        soft_label_values = [x / (self.n_dil_it+1) for x in range(self.n_dil_it, 0, -1)]

        # dilation
        gt_dil, gt_dil_bin = dilate_mask(gt_data_np, soft_label_values)

        # random holes in dilated area
        gt_holes, gt_holes_bin = random_holes(gt_data_np, gt_dil, gt_dil_bin)

        # post-processing
        gt_pp = post_processing(gt_data_np, gt_holes, gt_holes_bin, gt_dil)

        gt_t = F.to_tensor(gt_pp)
        rdict = {
            'gt': gt_t,
        }
        sample.update(rdict)

        return sample


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


class SliceFilter(mt_filters.SliceFilter):
    """This class extends the SliceFilter that already
    filters for empty labels and inputs. It will filter
    slices that has only zeros after cropping. To avoid
    feeding empty inputs into the network.
    """
    def __call__(self, sample):
        super_ret = super().__call__(sample)

        # Already filtered by base class
        if not super_ret:
            return super_ret

        # Filter slices where there are no values after cropping
        input_img = Image.fromarray(sample['input'], mode='F')
        input_cropped = F.center_crop(input_img, (128, 128))
        input_cropped = np.array(input_cropped)
        count = np.count_nonzero(input_cropped)

        if count <= 0:
            return False

        return True


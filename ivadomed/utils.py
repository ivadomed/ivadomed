import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes, binary_closing, generate_binary_structure

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
    :param dilation_factor: float, controls the number of dilation iterations.
                            For each individual lesion, the number of dilation iterations is computed as follows:
                                nb_it = int(round(dilation_factor * sqrt(lesion_area)))
                            If dilation_factor <= 0, then no dilation will be perfomed.
    """

    def __init__(self, dilation_factor):
        self.dil_factor = dilation_factor


    def dilate_lesion(arr_bin, arr_soft, label_values):
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

       return arr_bin, arr_soft


    def dilate_arr(self, arr, dil_factor):
        # init output arrays
        arr_bin_out = np.zeros(arr.shape, dtype=np.int)
        arr_soft_out = np.zeros(arr.shape, dtype=np.float)

        # loop across each axial slice
        for idx in range(arr.shape[0]):
            # identify each object
            arr_labeled, lb_nb = label(arr[idx].astype(np.int))

            # loop across each object of each axial slice
            arr_bin_lst, arr_soft_lst = [], []
            for obj_idx in range(1, lb_nb+1):
                arr_bin_obj = (arr_labeled == obj_idx).astype(np.int)
                arr_soft_obj = np.copy(arr_bin_obj_idx).astype(np.float)
                # compute the number of dilation iterations depending on the size of the lesion
                nb_it = int(dil_factor * math.sqrt(arr_obj.sum()))
                # values of the voxels added to the input mask
                soft_label_values = [x / (nb_it+1) for x in range(nb_it, 0, -1)]
                # dilate lesion
                arr_bin_dil, arr_soft_dil = dilate_lesion(arr_bin_obj, arr_soft_obj, soft_label_values)
                arr_bin_lst.append(arr_bin_dil)
                arr_soft_lst.append(arr_soft_dil)

            # sum dilated objects
            arr_bin_idx, arr_soft_idx = np.sum(arr_bin_lst), np.sum(arr_soft_lst)
            # clip values in case dilated voxels overlap
            arr_bin_clip, arr_soft_clip = np.clip(arr_bin_idx, a_max=1), np.clip(arr_soft_idx, a_max=1.0)

            # save result
            arr_bin_out[idx] = arr_bin_clip
            arr_soft_out[idx] = arr_soft_clip

        return arr_soft_out, arr_bin_out


    def random_holes(self, arr_in, arr_soft, arr_bin):
        arr_soft_out = np.copy(arr_soft)
        for idx in range(arr_in.shape[0]):
            # coordinates of the new voxels, i.e. the ones from the dilation
            new_voxels_xx, new_voxels_yy = np.where(np.logical_xor(arr_bin[idx], arr_in[idx]))
            nb_new_voxels = new_voxels_xx.shape[0]

            # ratio of voxels added to the input mask from the dilated mask
            new_voxel_ratio = random.random()
            # randomly select new voxel indexes to remove
            idx_to_remove = random.sample(range(nb_new_voxels),
                                            int(round(nb_new_voxels * (1 - new_voxel_ratio))))
            # set to zero the here-above randomly selected new voxels
            arr_soft_out[idx, new_voxels_xx[idx_to_remove], new_voxels_yy[idx_to_remove]] = 0.0

        arr_bin_out = (arr_soft_out > 0).astype(np.int)

        return arr_soft_out, arr_bin_out

    def post_processing(self, arr_in, arr_soft, arr_bin, arr_dil):
        struct = np.expand_dims(generate_binary_structure(2, 1), 0)  # to restrict operations along the two last dimensions

        # remove new object that are not connected to the input mask
        for idx in range(arr_bin.shape[0]):  # a loop is needed because "label" function restrictions
            arr_labeled, lb_nb = label(arr_bin[idx])
            connected_to_in = arr_labeled * arr_in[idx]
            for lb in range(1, lb_nb+1):
                if np.sum(connected_to_in == lb) == 0:
                    arr_soft[idx][arr_labeled == lb] = 0

        # binary closing
        arr_bin_closed = binary_closing((arr_soft > 0).astype(bool), structure=struct)
        # fill binary holes
        arr_bin_filled = binary_fill_holes(arr_bin_closed, structure=struct)

        # recover the soft-value assigned to the filled-holes
        arr_soft_out = arr_bin_filled * arr_dil

        return arr_soft_out

    def __call__(self, sample):
        if self.dil_factor > 0:
            gt_data = sample['gt']
            gt_data_np = gt_data.numpy()

            # index of samples where ones
            idx_ones = np.unique(np.where(gt_data_np)[0])

            # dilation
            gt_dil, gt_dil_bin = self.dilate_arr(gt_data_np[idx_ones,0], self.dil_factor)

            # random holes in dilated area
            gt_holes, gt_holes_bin = self.random_holes(gt_data_np[idx_ones,0], gt_dil, gt_dil_bin)

            # post-processing
            gt_pp = self.post_processing(gt_data_np[idx_ones,0], gt_holes, gt_holes_bin, gt_dil)
            gt_out = gt_data_np.astype(np.float64)
            gt_out[idx_ones,0] = gt_pp

            gt_t = F.to_tensor(gt_out[:,0])  # input of F.to_tensor needs to have 3 dimensions
            gt_t = gt_t.permute(1, 2, 0)
            rdict = {
                'gt': gt_t.unsqueeze_(1),  # add dimension back
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


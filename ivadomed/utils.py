import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import defaultdict

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
        input_cropped = transforms.functional.center_crop(input_img, (128, 128))
        input_cropped = np.array(input_cropped)
        count = np.count_nonzero(input_cropped)

        if count <= 0:
            return False

        return True


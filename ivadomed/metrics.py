from collections import defaultdict

from scipy import spatial
import numpy as np


class MetricManager(object):
    def __init__(self, metric_fns):
        self.metric_fns = metric_fns
        self.result_dict = defaultdict(float)
        self.num_samples = 0

    def __call__(self, prediction, ground_truth):
        self.num_samples += len(prediction)
        for metric_fn in self.metric_fns:
            for p, gt in zip(prediction, ground_truth):
                res = metric_fn(p, gt)
                dict_key = metric_fn.__name__
                self.result_dict[dict_key] += res

    def get_results(self):
        res_dict = {}
        for key, val in self.result_dict.items():
            res_dict[key] = val / self.num_samples
        return res_dict

    def reset(self):
        self.num_samples = 0
        self.result_dict = defaultdict(float)


def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN


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


def jaccard_score(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    return (1 - spatial.distance.jaccard(pflat, gflat)) * 100.0


def hausdorff_2D_score(prediction, groundtruth):
    return spatial.distance.directed_hausdorff(prediction, groundtruth)[0]


def hausdorff_3D_score(prediction, groundtruth):
    if len(prediction.shape) == 3:
        mean_hansdorff = 0
        for idx in range(prediction.shape[1]):
            pred = prediction[:, idx, :]
            gt = groundtruth[:, idx, :]
            mean_hansdorff += hausdorff_2D_score(pred, gt)
        mean_hansdorff = mean_hansdorff / prediction.shape[1]
        return mean_hansdorff

    return hausdorff_2D_score(prediction, groundtruth)


def precision_score(prediction, groundtruth, err_value=0.0):
    # PPV
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return err_value

    precision = np.divide(TP, TP + FP)
    return precision * 100.0


def recall_score(prediction, groundtruth, err_value=0.0):
    # TPR, sensitivity
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return err_value
    TPR = np.divide(TP, TP + FN)
    return TPR * 100.0


def specificity_score(prediction, groundtruth, err_value=0.0):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TN + FP) <= 0.0:
        return err_value
    TNR = np.divide(TN, TN + FP)
    return TNR * 100.0


def intersection_over_union(prediction, groundtruth, err_value=0.0):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return err_value
    return TP / (TP + FP + FN) * 100.0


def accuracy_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0

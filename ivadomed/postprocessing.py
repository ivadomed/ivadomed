# Deals with postprocessing on generated segmentation.

import numpy as np
from scipy.ndimage.measurements import label


def threshold_predictions(predictions, thr=0.5):
    """Threshold predictions.

    Threshold a soft (ie not binary) array of predictions given a threshold value, and returns
    a binary array.

    Args:
        predictions (array): array to binarise.
        thr (float): Threshold value: voxels with a value < to thr are assigned 0 as value, 1
            otherwise.
    Returns:
        array: Array containing only zeros or ones.

    """
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

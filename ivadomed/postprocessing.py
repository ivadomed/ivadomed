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


def keep_largest_object(predictions):
    """Keep the largest connect object.

    Keep the largest connected object per z_slice and fill little holes.
    Note: This function only works for binary segmentation.
    :param z_slice: int 2d-array: Input 2d segmentation
    :return: z_slice: int 2d-array: Processed 2d segmentation
    """
    assert predictions.dtype == np.dtype('int')
    # Find number of closed objects using skimage "label"
    labeled_obj, num_obj = label(predictions)
    # If more than one object is found, keep the largest one
    if num_obj > 1:
        # Keep the largest object
            predictions[np.where(labeled_obj != (np.bincount(labeled_obj.flat)[1:].argmax() + 1))] = 0
    return predictions

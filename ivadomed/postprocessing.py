# Deals with postprocessing on generated segmentation.

import numpy as np
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_fill_holes


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

    Keep the largest connected object from the input array (2 or 3D).
    Note: This function only works for binary segmentation.

    Args:
        predictions (array): Input 2 or 3D binary segmentation.
    Returns:
        array: processed segmentation.

    """
    assert predictions.dtype == np.dtype('int')
    # Find number of closed objects using skimage "label"
    labeled_obj, num_obj = label(predictions)
    # If more than one object is found, keep the largest one
    if num_obj > 1:
        # Keep the largest object
            predictions[np.where(labeled_obj != (np.bincount(labeled_obj.flat)[1:].argmax() + 1))] = 0
    return predictions


def keep_largest_object_per_slice(predictions, axis=2):
    """Keep the largest connect object per 2d slice.

    Keep the largest connected object for each 2D slice of an input array along a given axis.
    Note: This function only works for binary segmentation.

    Args:
        predictions (array): Input binary segmentation.
        axis (int): 2D slices are extracted along this axis.
    Returns:
        array: processed segmentation.

    """
    assert predictions.dtype == np.dtype('int')
    # Split the 3D input array as a list of slice along axis
    list_preds_in = np.split(predictions, predictions.shape[axis], axis=axis)
    # Init list of processed slices
    list_preds_out = []
    # Loop across the slices along the given axis
    for idx in range(len(list_preds_in)):
        slice_processed = keep_largest_object(list_preds_in[idx])
        preds_out.append(slice_processed)
        print(slice_processed.shape, list_preds_in[idx].shape)
    print(predictions.shape, np.stack(list_preds_out, axis=axis).shape)
    return np.stack(list_preds_out, axis=axis)


def fill_holes(predictions, structure=(3,3,3)):
    """Fill holes in the predictions.

    Fill holes in the predictions using a given structuring element.

    Args:
        predictions (array): Input binary segmentation, 2 or 3D.
        structure (tuple of integers): Structuring element, number of ints equals
            number of dimensions in the input array.
    Returns:
        array: processed segmentation.

    """
    assert predictions.dtype == np.dtype('int')
    assert len(structure) == predictions.shape)
    return binary_fill_holes(predictions, structure=np.ones(structure)).astype(np.int)

def mask_soft_predictions(predictions_soft, predictions_bin):
    assert predictions_soft.shape == predictions_bin.shape
    # Check if predictions_bin only contains 0s or 1s
    assert predictions_bin.dtype == np.dtype('int')
    assert np.array_equal(predictions_bin, predictions_bin.astype(bool))
    return predictions_soft * predictions_bin

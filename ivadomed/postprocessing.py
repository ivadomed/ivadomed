# Deals with postprocessing on generated segmentation.

import functools
import numpy as np
import nibabel as nib
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_fill_holes


def nifti_capable(wrapped):
    @functools.wraps(wrapped)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, nib.Nifti1Image):
            return nib.Nifti1Image(wrapper(np.asanyarray(data.dataobj), *args, **kwargs), data.affine)
        return wrapped(data, *args, **kwargs)
    return wrapper


@nifti_capable
def threshold_predictions(predictions, thr=0.5):
    """
    Threshold a soft (ie not binary) array of predictions given a threshold value, and returns
    a binary array.

    Args:
        predictions (array or nibabel object): Image to binarise.
        thr (float): Threshold value: voxels with a value < to thr are assigned 0 as value, 1
            otherwise.
    Returns:
        array: Array or nibabel (same object as the input) containing only zeros or ones. Output type is int.
    """
    thresholded_preds = np.copy(predictions)[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds.astype(np.int)


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


def keep_largest_object_nib(nib_predictions):
    """Keep the largest connect object from nibabel image.

    Keep the largest connected object from the input image (2 or 3D).
    Note: This function only works for binary segmentation.
    Note: This function calls keep_largest_object().

    Args:
        nib_predictions (nibabelObject): Input image, 2 or 3D, binary segmentation.
    Returns:
        nibabelObject: processed segmentation.

    """
    data = nib_predictions.get_fdata()
    data_out = keep_largest_object(predictions=data)
    return nib.Nifti1Image(data_out, nib_predictions.affine)


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
        slice_processed = keep_largest_object(np.squeeze(list_preds_in[idx], axis=axis))
        list_preds_out.append(slice_processed)
    return np.stack(list_preds_out, axis=axis)


def keep_largest_object_per_slice_nib(nib_predictions, axis=2):
    """Keep the largest connect object from nibabel image per 2d slice.

    Keep the largest connected object for each 2D slice of an input image along a given axis.
    Note: This function only works for binary segmentation.
    Note: This function calls keep_largest_object_per_slice().

    Args:
        nib_predictions (nibabelObject): Input image, 2 or 3D, binary segmentation.
        axis (int): 2D slices are extracted along this axis.
    Returns:
        nibabelObject: processed segmentation.

    """
    data = nib_predictions.get_fdata()
    data_out = keep_largest_object_per_slice(predictions=data, axis=axis)
    return nib.Nifti1Image(data_out, nib_predictions.affine)


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
    assert len(structure) == len(predictions.shape)
    return binary_fill_holes(predictions, structure=np.ones(structure)).astype(np.int)


def fill_holes_nib(nib_predictions, structure=(3,3,3)):
    """Fill holes in the predictions.

    Fill holes in the nibabel predictions using a given structuring element.

    Args:
        nib_predictions (nibabelObject): Input image, binary segmentation, 2 or 3D.
        structure (tuple of integers): Structuring element, number of ints equals
            number of dimensions in the input image.
    Returns:
        nibabelObject: processed segmentation.

    """
    data = nib_predictions.get_fdata()
    data_out = fill_holes(predictions=data, structure=structure)
    return nib.Nifti1Image(data_out, nib_predictions.affine)


def mask_predictions(predictions, mask_binary):
    """Mask soft predictions with binary mask.

    Mask predictions (e.g. soft predictions) using a binary mask (e.g. ROI, hard predictions).

    Args:
        predictions (array): array to mask.
        mask_binary (array): array with the same shape as predictions, containing only zeros or ones.
    Returns:
        array: processed segmentation.

    """
    assert predictions.shape == mask_binary.shape
    # Check if predictions_bin only contains 0s or 1s
    assert mask_binary.dtype == np.dtype('int')
    assert np.array_equal(mask_binary, mask_binary.astype(bool))
    return predictions * mask_binary


def mask_predictions_nib(nib_predictions, nib_mask_binary):
    """Mask soft predictions with binary mask.

    Mask nibabel predictions (e.g. soft predictions) using a nibabel binary mask (e.g. ROI, hard predictions).

    Args:
        nib_predictions (nibabelObject): nibabel image to mask.
        nib_mask_binary (nibabelObject): nibabel image with the same shape as nib_predictions, containing only zeros or ones.
    Returns:
        nibabelObject: processed segmentation.

    """
    data = nib_predictions.get_fdata()
    data_mask = nib_mask_binary.get_fdata()
    data_out = mask_predictions(predictions=data, mask_binary=data_mask)
    return nib.Nifti1Image(data_out, nib_predictions.affine)

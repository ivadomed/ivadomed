# Deals with postprocessing on generated segmentation.

import functools
import numpy as np
import nibabel as nib
import pydensecrf.densecrf as dcrf
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_fill_holes


def nifti_capable(wrapped):
    @functools.wraps(wrapped)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, nib.Nifti1Image):
            return nib.Nifti1Image(wrapper(np.copy(np.asanyarray(data.dataobj)), *args, **kwargs), data.affine)
        return wrapped(data, *args, **kwargs)

    return wrapper


def binarize_with_low_threshold(wrapped):
    @functools.wraps(wrapped)
    def wrapper(data, *args, **kwargs):
        if not np.array_equal(data, data.astype(bool)):
            return mask_predictions(data, wrapper(threshold_predictions(data, thr=0.001), *args, **kwargs))
        return wrapped(data, *args, **kwargs)

    return wrapper


def multilabel_capable(wrapped):
    @functools.wraps(wrapped)
    def wrapper(data, *args, **kwargs):
        if len(data.shape) == 4:
            label_list = []
            for i in range(data.shape[-1]):
                out_data = wrapped(data[..., i], *args, **kwargs)
                label_list.append(out_data)
            return np.array(label_list).transpose((1, 2, 3, 0))
        return wrapped(data, *args, **kwargs)

    return wrapper


@nifti_capable
def threshold_predictions(predictions, thr=0.5):
    """
    Threshold a soft (ie not binary) array of predictions given a threshold value, and returns
    a binary array.

    Args:
        predictions (array or nibabel object): Image to binarize.
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


@nifti_capable
@binarize_with_low_threshold
def keep_largest_object(predictions):
    """
    Keep the largest connected object from the input array (2D or 3D).

    Args:
        predictions (array or nibabel object): Input segmentation. Image could be 2D or 3D.
    Returns:
        Array or nibabel (same object as the input).
    """
    # Find number of closed objects using skimage "label"
    labeled_obj, num_obj = label(np.copy(predictions))
    # If more than one object is found, keep the largest one
    if num_obj > 1:
        # Keep the largest object
        predictions[np.where(labeled_obj != (np.bincount(labeled_obj.flat)[1:].argmax() + 1))] = 0
    return predictions


@nifti_capable
def keep_largest_object_per_slice(predictions, axis=2):
    """
    Keep the largest connected object for each 2D slice, along a specified axis.

    Args:
        predictions (array or nibabel object): Input segmentation. Image could be 2D or 3D.
        axis (int): 2D slices are extracted along this axis.
    Returns:
        Array or nibabel (same object as the input).
    """
    # Split the 3D input array as a list of slice along axis
    list_preds_in = np.split(predictions, predictions.shape[axis], axis=axis)
    # Init list of processed slices
    list_preds_out = []
    # Loop across the slices along the given axis
    for idx in range(len(list_preds_in)):
        slice_processed = keep_largest_object(np.squeeze(list_preds_in[idx], axis=axis))
        list_preds_out.append(slice_processed)
    return np.stack(list_preds_out, axis=axis)


@nifti_capable
@multilabel_capable
def fill_holes(predictions, structure=(3, 3, 3)):
    """
    Fill holes in the predictions using a given structuring element.
    Note: This function only works for binary segmentation.

    Args:
        predictions (array or nibabel object): Input binary segmentation. Image could be 2D or 3D.
        structure (tuple of integers): Structuring element, number of ints equals
            number of dimensions in the input array.
    Returns:
        Array or nibabel (same object as the input). Output type is int.
    """
    assert np.array_equal(predictions, predictions.astype(bool))
    assert len(structure) == len(predictions.shape)
    return binary_fill_holes(predictions, structure=np.ones(structure)).astype(np.int)


@nifti_capable
def mask_predictions(predictions, mask_binary):
    """
    Mask predictions using a binary mask: sets everything outside the mask to zero.

    Args:
        predictions (array or nibabel object): Input binary segmentation. Image could be 2D or 3D.
        mask_binary (array): array with the same shape as predictions, containing only zeros or ones.
    Returns:
        Array or nibabel (same object as the input).
    """
    assert predictions.shape == mask_binary.shape
    assert np.array_equal(mask_binary, mask_binary.astype(bool))
    return predictions * mask_binary


def apply_crf(predictions, image, n_iterations=5, eps=1e-6):
    """
    Apply Conditional Random Fields to the soft predictions. 2D inputs, with shape: n_label, height, width.

    Args:
        predictions (np.array): Input 2D soft segmentation.
        image (np.array): Input 2D image.
        n_iterations (int):
        eps (float): To avoid log(0), need to clip 0 probabilities to a positive value
    Returns:
        Array.
    """
    # Get data shape
    height, width, n_label = predictions.shape

    # Init DenseCRF
    d = dcrf.DenseCRF2D(width, height, n_label)

    # UNARY
    # Transpose axes
    predictions.transpose(2, 0, 1)
    # Clip 0 probabilities as unary is negative log probabilities
    predictions = np.clip(predictions, eps, 1.0)
    # Get Unary
    unary_potentials = -np.log(predictions).astype(np.float32)
    # Flatten
    unary_potentials.reshape([n_label, -1])
    # Set Unary potentials
    d.setUnaryEnergy(np.ascontiguousarray(unary_potentials))

    # PAIRWISE
    x_y_sd = (10, 10)
    scale_feature = 0.01  # Image feature scaling factor per channel
    spatial_potential_strength = 10
    feature_potential_strength = 10
    # Spatial prior: enforces more spatially consistent segmentation
    spatial_prior = dcrf.create_pairwise_gaussian(sdims=x_y_sd, shape=(width, height))
    d.addPairwiseEnergy(spatial_prior, compat=spatial_potential_strength)
    # Feature prior: voxels with either a similar features are likely to belong to the same class
    # TODO: add uncertainty as feature --> change chdim
    feature_prior = dcrf.create_pairwise_bilateral(sdims=x_y_sd, schan=scale_feature, img=image, chdim=-1)
    d.addPairwiseEnergy(feature_prior, compat=feature_potential_strength)

    # INFERENCE
    Q = d.inference(n_iterations)
    # Get MAP predictions
    map = np.argmax(Q, axis=0).reshape((height, width))

    """
    # STEP BY STEP INFERENCE
    Q, tmp1, tmp2 = d.startInference()
    for i in range(n_iterations):
        print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)
    kl = d.klDivergence(Q) / (height * width)
    map = np.argmax(Q, axis=0).reshape((height, width))
    """
    return map

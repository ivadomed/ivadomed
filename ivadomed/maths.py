import nibabel as nib
import numpy as np
import os
import scipy
import ivadomed.transforms as imed_transforms


def gaussian_kernel(kernlen=10):
    """
    Create a 2D gaussian kernel with user-defined size.

    Args:
        kernlen (int): size of kernel

    Returns:
        ndarray: a 2D array of size (kernlen,kernlen)
    """

    x = np.linspace(-1, 1, kernlen + 1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return imed_transforms.rescale_values_array(kern2d / kern2d.sum())


def heatmap_generation(image, kernel_size):
    """
    Generate heatmap from image containing sing voxel label using
    convolution with gaussian kernel
    Args:
        image (ndarray): 2D array containing single voxel label
        kernel_size (int): size of gaussian kernel

    Returns:
        ndarray: 2D array heatmap matching the label.

    """
    kernel = gaussian_kernel(kernel_size)
    map = scipy.signal.convolve(image, kernel, mode='same')
    return imed_transforms.rescale_values_array(map)


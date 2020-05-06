#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for ivadomed.postprocessing


import pytest
import numpy as np
from math import isclose

import torch

from ivadomed.transforms import HistogramClipping, RandomShiftIntensity, NumpyToTensor, Resample, rescale_array


def create_test_image_2d(width, height, num_modalities, noise_max=10.0, num_objs=1, rad_max=30, num_seg_classes=1):
    """Create test image.

    Create test image and its segmentation with a given number of objects, classes, and maximum radius.

    Args:
        width (int): width image
        height (int): height image
        num_modalities (int): number of modalities
        noise_max (float): noise from the uniform distribution [0,noise_max)
        num_objs (int): number of objects
        rad_max (int): maximum radius of objects
        num_seg_classes (int): number of classes
    Return:
        list, list: image and segmentation, list of num_modalities elements of shape (width, height).

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/data/synthetic.py#L17
    """
    assert num_modalities >= 1

    image = np.zeros((width, height))

    for i in range(num_objs):
        x = np.random.randint(rad_max, width - rad_max)
        y = np.random.randint(rad_max, height - rad_max)
        rad = np.random.randint(5, rad_max)
        spy, spx = np.ogrid[-x:width - x, -y:height - y]
        circle = (spx * spx + spy * spy) <= rad * rad

        if num_seg_classes > 1:
            image[circle] = np.ceil(np.random.random() * num_seg_classes)
        else:
            image[circle] = np.random.random() * 0.5 + 0.5

    seg = np.ceil(image).astype(np.int32)

    list_im, list_seg = [], []
    for _ in range(num_modalities):
        norm = np.random.uniform(0, num_seg_classes * noise_max, size=image.shape)
        noisy_image = rescale_array(np.maximum(image, norm))

        list_im.append(noisy_image)
        list_seg.append(seg)

    return list_im, list_seg


@pytest.mark.parametrize('im_seg', [create_test_image_2d(100, 100, 1),
                                    create_test_image_2d(100, 100, 3)])
def test_HistogramClipping(im_seg):
    im, _ = im_seg
    # Transform
    transform = HistogramClipping()
    # Apply Transform
    metadata = [{} for _ in im] if isinstance(im, list) else {}
    do_im, _ = transform(sample=im, metadata=metadata)
    # Check result has the same number of modalities
    assert len(do_im) == len(im)
    # Check clipping
    min_percentile = transform.min_percentile
    max_percentile = transform.max_percentile
    for i, r in zip(im, do_im):
        assert isclose(np.min(r), np.percentile(i, min_percentile), rel_tol=1e-02)
        assert isclose(np.max(r), np.percentile(i, max_percentile), rel_tol=1e-02)


@pytest.mark.parametrize('im_seg', [create_test_image_2d(100, 100, 1),
                                    create_test_image_2d(100, 100, 3)])
def test_RandomShiftIntensity(im_seg):
    im, _ = im_seg
    # Transform
    transform = RandomShiftIntensity(shift_range=[0., 10.])

    # Apply Do Transform
    metadata_in = [{} for _ in im] if isinstance(im, list) else {}
    do_im, do_metadata = transform(sample=im, metadata=metadata_in)
    # Check result has the same number of modalities
    assert len(do_im) == len(im)
    # Check metadata update
    assert all('offset' in m for m in do_metadata)
    # Check shifting
    for idx, i in enumerate(im):
        assert isclose(np.max(do_im[idx]-i), do_metadata[idx]['offset'], rel_tol=1e-02)

    # Apply Undo Transform
    undo_im, undo_metadata = transform.undo_transform(sample=do_im, metadata=do_metadata)
    # Check result has the same number of modalities
    assert len(undo_im) == len(im)
    # Check undo
    for idx, i in enumerate(im):
        assert np.allclose(undo_im[idx], i, rtol=1e-02)


@pytest.mark.parametrize('im_seg', [create_test_image_2d(100, 100, 1),
                                    create_test_image_2d(100, 100, 3)])
def test_NumpyToTensor(im_seg):
    im, seg = im_seg
    metadata_in = [{} for _ in im] if isinstance(im, list) else {}

    # Transform
    transform = NumpyToTensor()

    for im_cur in [im, seg]:
        # Numpy to Tensor
        do_im, do_metadata = transform(sample=im_cur, metadata=metadata_in)
        for idx, i in enumerate(do_im):
            assert torch.is_tensor(i)

        # Tensor to Numpy
        undo_im, undo_metadata = transform.undo_transform(sample=do_im, metadata=do_metadata)
        for idx, i in enumerate(undo_im):
            assert isinstance(i, np.ndarray)
            assert np.array_equal(i, im_cur[idx])
            assert i.dtype == im_cur[idx].dtype


#@pytest.mark.parametrize('im_seg', (create_test_image_2d(80, 100, 1),
#                                    create_test_image_2d(100, 80, 3)))
#@pytest.mark.parametrize('resample_transform', (Resample(0.5, 1.0, interpolation_order=2),
#                                                Resample(1.0, 0.5, interpolation_order=2)))
#@pytest.mark.parametrize('native_shape', ((90, 110),
#                                               (110, 90)))
@pytest.mark.parametrize('im_seg', [create_test_image_2d(80, 100, 1),
                                    create_test_image_2d(100, 80, 1)])
@pytest.mark.parametrize('resample_transform', [Resample(0.5, 1.0, interpolation_order=2),
                                                Resample(1.0, 0.5, interpolation_order=2)])
@pytest.mark.parametrize('native_resolution', [(0.9, 1.0),
                                               (1.0, 0.9)])
def test_Resample(im_seg, resample_transform, native_resolution):
    im, seg = im_seg
    metadata_ = {'zooms': native_resolution, 'data_shape': im[0].shape}
    metadata_in = [metadata_ for _ in im] if isinstance(im, list) else {}

    # Resample input data
    do_im, do_metadata = resample_transform(sample=im, metadata=metadata_in)

    # TODO: Check expected resolution

    # Undo Resample on input data
    undo_im, _ = resample_transform.undo_transform(sample=do_im, metadata=do_metadata)

    # Check data content and data shape between input data and undo
    for idx, i in enumerate(im):
        assert i.shape == undo_im[idx].shape
        # TODO: Matplotlib
        print(np.max(undo_im[idx] - i), np.min(undo_im[idx] - i))
        #assert np.allclose(undo_im[idx], i, rtol=1e-02)

    # TODO: Check dtype
    # TODO: Check dtype for seg
    # TODO: Check data consistency / interpolation mode for seg
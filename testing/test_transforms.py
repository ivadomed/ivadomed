#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for ivadomed.postprocessing


import pytest
import numpy as np
from math import isclose

from ivadomed.transforms import HistogramClipping, RandomShiftIntensity, rescale_array


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


@pytest.mark.parametrize('im_seg', (create_test_image_2d(100, 100, 1),
                                    create_test_image_2d(100, 100, 3)))
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


@pytest.mark.parametrize('im_seg', (create_test_image_2d(100, 100, 1),
                                    create_test_image_2d(100, 100, 3)))
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

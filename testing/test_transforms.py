#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for ivadomed.postprocessing


import pytest
import numpy as np
from math import isclose

import torch

from ivadomed.transforms import ROICrop2D, CenterCrop2D, NormalizeInstance, HistogramClipping, RandomShiftIntensity, NumpyToTensor, Resample, rescale_array
from ivadomed.metrics import dice_score, mse

DEBUGGING = False
if DEBUGGING:
    from testing.utils import plot_transformed_sample


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


@pytest.mark.parametrize('im_seg', [create_test_image_2d(80, 100, 1),
                                    create_test_image_2d(100, 80, 1)])
@pytest.mark.parametrize('resample_transform', [Resample(0.8, 1.0, interpolation_order=2),
                                                Resample(1.0, 0.8, interpolation_order=2)])
@pytest.mark.parametrize('native_resolution', [(0.9, 1.0),
                                               (1.0, 0.9)])
def test_Resample(im_seg, resample_transform, native_resolution):
    im, seg = im_seg
    metadata_ = {'zooms': native_resolution, 'data_shape': im[0].shape}
    metadata_in = [metadata_ for _ in im] if isinstance(im, list) else {}

    # Resample input data
    do_im, do_metadata = resample_transform(sample=im, metadata=metadata_in)
    # Undo Resample on input data
    undo_im, _ = resample_transform.undo_transform(sample=do_im, metadata=do_metadata)

    # Resampler for label data
    resample_transform.interpolation_order = 0
    # Resample label data
    do_seg, do_metadata = resample_transform(sample=seg, metadata=metadata_in)
    # Undo Resample on label data
    undo_seg, _ = resample_transform.undo_transform(sample=do_seg, metadata=do_metadata)

    # Check data content and data shape between input data and undo
    for idx, i in enumerate(im):
        # Check shapes
        assert i.shape == undo_im[idx].shape == seg[idx].shape == undo_seg[idx].shape
        assert do_seg[idx].shape == do_im[idx].shape
        # Check data type
        assert i.dtype == do_im[idx].dtype == undo_im[idx].dtype
        assert seg[idx].dtype == do_seg[idx].dtype == undo_seg[idx].dtype
        # Plot for debugging
        if DEBUGGING:
            plot_transformed_sample(seg[idx], undo_seg[idx])
        # Data consistency
        assert dice_score(undo_seg[idx], seg[idx]) > 0.8
        assert mse(undo_im[idx], im[idx]) < 1e-1

@pytest.mark.parametrize('im_seg', [create_test_image_2d(100, 100, 1),
                                    create_test_image_2d(100, 100, 3)])
def test_NormalizeInstance(im_seg):
    im, seg = im_seg
    metadata_in = [{} for _ in im] if isinstance(im, list) else {}

    # Transform on Numpy
    transform = NormalizeInstance()
    do_im, _ = transform(im.copy(), metadata_in)
    # Check normalization
    for i in do_im:
        assert abs(np.mean(i) - 0.0) <= 1e-2
        assert abs(np.std(i) - 1.0) <= 1e-2

    # Transform on Tensor
    tensor, metadata_tensor = NumpyToTensor()(im, metadata_in)
    do_tensor, _ = transform(tensor, metadata_tensor)
    # Check normalization
    for t in do_tensor:
        assert abs(t.mean() - 0.0) <= 1e-2
        assert abs(t.std() - 1.0) <= 1e-2


@pytest.mark.parametrize('im_seg', [create_test_image_2d(100, 100, 1)])
@pytest.mark.parametrize('crop_transform', [CenterCrop2D((80, 60))])
                                            #CenterCrop2D((60, 80)),
                                            #ROICrop2D((80, 60)),
                                            #ROICrop2D((60, 80))])
def test_Crop2D(im_seg, crop_transform):
    im, seg = im_seg
    metadata_ = {'data_shape': im[0].shape}
    metadata_in = [metadata_ for _ in im] if isinstance(im, list) else {}

    if crop_transform.__class__.__name__ == "ROICrop2D":
        _, metadata_in = crop_transform(seg, metadata_in)
        for metadata in metadata_in:
            assert "crop_params" in metadata

    # Apply transform
    do_im, do_metadata = crop_transform(im, metadata_in)
    do_seg, do_seg_metadata = crop_transform(seg, metadata_in)

    # Loop and check
    for idx, i in enumerate(im):
        # Check data shape
        assert do_im[idx].shape == crop_transform.size
        assert do_seg[idx].shape == crop_transform.size
        # Check metadata
        assert do_metadata[idx]['crop_params'] == do_seg_metadata[idx]['crop_params']

    # Apply undo transform
    undo_im, _ = crop_transform.undo_transform(do_im, do_metadata)
    undo_seg, _ = crop_transform.undo_transform(do_seg, do_seg_metadata)

    # Loop and check
    for idx, i in enumerate(im):
        # Check data shape
        assert undo_im[idx].shape == i.shape
        assert undo_seg[idx].shape == seg[idx].shape
        # Check data type
        assert do_im[idx].dtype == undo_im[idx].dtype == i.dtype
        assert do_seg[idx].dtype == undo_seg[idx].dtype == seg[idx].dtype
        # Check data consistency
        fh, fw, _, _ = do_metadata[idx]['crop_params']
        th, tw = crop_transform.size
        assert np.array_equal(i[fh:fh+th, fw:fw+tw], undo_im[idx][fh:fh+th, fw:fw+tw])
        assert np.array_equal(seg[idx][fh:fh+th, fw:fw+tw], undo_seg[idx][fh:fh+th, fw:fw+tw])
        # Plot for debugging
        if DEBUGGING:
            plot_transformed_sample(seg[idx], undo_seg[idx])
            plot_transformed_sample(i, undo_im[idx])

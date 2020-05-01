#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for ivadomed.postprocessing


import pytest
import numpy as np

def create_test_image_2d(width, height, num_objs=1, rad_max=30, num_seg_classes=1):
    """Create test image.

    Create test image and its label with a given number of objects, classes, and maximum radius.

    Args:
        width (int): width image
        height (int): height image
        num_objs (int): number of objects
        rad_max (int): maximum radius of objects
        num_seg_classes (int): number of classes
    Return:
        np.array, np.array: image and label

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/data/synthetic.py#L17
    """
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

    return image, seg
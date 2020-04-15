#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for ivadomed.postprocessing


import pytest
import numpy as np
import nibabel as nib

import ivadomed.postprocessing as postproc


@pytest.fixture(scope="session")
def nii_seg(size_arr=(9, 9, 9), pixdim=(1, 1, 1), dtype=np.float64, orientation='LPI', shape='rectangle',
            radius_RL=2.0, radius_AP=1.0, zeroslice=[]):
    """Create a dummy nibabel object with a ellipse or rectangle of ones running from top to bottom in the 3rd
    dimension. Add a voxel with value '1' outside the main object to test the keep largest object algorithms.
    :param size_arr: tuple: (nx, ny, nz)
    :param pixdim: tuple: (px, py, pz)
    :param dtype: Numpy dtype.
    :param orientation: Orientation of the image. Default: LPI
    :param shape: {'rectangle', 'ellipse'}
    :param radius_RL: float: 1st radius. With a, b = 50.0, 30.0 (in mm), theoretical CSA of ellipse is 4712.4
    :param radius_AP: float: 2nd radius
    :param zeroslice: list int: zero all slices listed in this param
    :param debug: Write temp files for debug
    :return: nibabel: Image object
    """
    # Create a 3d array, with dimensions corresponding to x: RL, y: AP, z: IS
    nx, ny, nz = [int(size_arr[i] * pixdim[i]) for i in range(3)]
    data = np.zeros((nx, ny, nz), dtype)
    xx, yy = np.mgrid[:nx, :ny]
    # loop across slices and add object
    for iz in range(nz):
        if shape == 'rectangle':  # theoretical CSA: (a*2+1)(b*2+1)
            data[:, :, iz] = ((abs(xx - nx / 2) <= radius_RL) & (abs(yy - ny / 2) <= radius_AP)) * 1
        if shape == 'ellipse':
            data[:, :, iz] = (((xx - nx / 2) / radius_RL) ** 2 + ((yy - ny / 2) / radius_AP) ** 2 <= 1) * 1

    # Zero specified slices
    if zeroslice is not []:
        data[:, :, zeroslice] = 0

    # Add voxel outside of the main object
    data[1, 1, 1] = 1

    # Apply Gaussian filter (to get soft seg)
    # TODO

    # Create nibabel object
    affine = np.eye(4)
    # for i in range(3):
    #     affine[i][i] = 1  # in [mm]
    nii = nib.nifti1.Nifti1Image(data, affine)


    # Change orientation
    # TODO

    return nii


def test_threshold(nii_seg):
    # input array
    arr_seg_proc = postproc.threshold_predictions(np.asanyarray(nii_seg.dataobj))
    assert isinstance(arr_seg_proc, np.ndarray)
    # input nibabel
    nii_seg_proc = postproc.threshold_predictions(nii_seg)
    assert isinstance(nii_seg_proc, nib.nifti1.Nifti1Image)


def test_keep_largest_object(nii_seg):
    arr_seg_proc = postproc.keep_largest_object(np.asanyarray(nii_seg.dataobj))
    assert isinstance(arr_seg_proc, np.ndarray)
    assert arr_seg_proc[1, 1, 1] == 0

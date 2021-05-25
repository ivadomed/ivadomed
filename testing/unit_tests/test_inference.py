from ivadomed.inference import pred_to_nib
import logging
import pytest
import os
import numpy as np
import nibabel as nib
from unit_tests.t_utils import remove_tmp_dir, create_tmp_dir, __data_testing_dir__, __tmp_dir__
logger = logging.getLogger(__name__)

__output_dir__ = os.path.join(__tmp_dir__, "output_inference")


def setup_function():
    create_tmp_dir()
    os.mkdir(__output_dir__)


@pytest.mark.parametrize('kernel_dim', ['2d'])
@pytest.mark.parametrize('slice_axis', [0, 1, 2])
@pytest.mark.parametrize('qform_affine', [
    None,
    np.array([
        [2., 0., 0., 0.],
        [0., 3., 0., 0.],
        [0., 0., 4., 0.],
        [0., 0., 0., 1.]
    ])
])
@pytest.mark.parametrize('qform_code', [None, 0, 1, 2, 3, 4])
def test_pred_to_nib(kernel_dim, slice_axis, qform_affine, qform_code):
    arr = np.array(
        [[[0, 1.0, 1.0],
          [0, 0, 1.0]],
         [[0, 1.0, 1.0],
         [0, 1.0, 0]]])
    img_original = nib.Nifti1Image(
        dataobj=arr,
        affine=None,
        header=None
    )
    img_original.set_qform(
        qform_affine, qform_code
    )
    nib.save(img_original, os.path.join(__data_testing_dir__, "image_ref.nii.gz"))
    data_lst = [
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([[1.0, 0.0], [1.0, 1.0]]),
        np.array([[1.0, 1.0], [1.0, 0.0]])
    ]
    z_lst = [0, 1, 2]
    nib_pred = pred_to_nib(
        data_lst,
        z_lst,
        fname_ref=os.path.join(__data_testing_dir__, "image_ref.nii.gz"),
        fname_out=os.path.join(__output_dir__, "nib_image.nii.gz"),
        slice_axis=slice_axis,
        debug=False,
        kernel_dim=kernel_dim,
        bin_thr=0.5,
        discard_noise=True,
        postprocessing=None
    )
    assert os.path.exists(os.path.join(__output_dir__, "nib_image.nii.gz"))
    qform = nib_pred.get_qform(coded=True)
    if qform_affine is None:
        if qform_code is None:
            assert qform[0] is None
        else:
            affine = np.array([
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]
                ])
            assert (qform[0] == affine).all()
    else:
        assert (qform[0] == qform_affine).all()
    assert qform[1] == qform_code


def teardown_function():
    remove_tmp_dir()

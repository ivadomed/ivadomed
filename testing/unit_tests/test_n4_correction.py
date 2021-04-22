import pytest
import os
from ivadomed.scripts import n4_correction
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, __tmp_dir__
from testing.common_testing_util import remove_tmp_dir


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('test_lst', ['sub-unf01', 'sub-unf02', 'sub-unf03'])
def test_n4_correction(test_lst):
    path = os.path.join(__data_testing_dir__, test_lst, "anat")
    new_path = os.path.join(__tmp_dir__, "n4_corrected", test_lst, "anat")
    os.makedirs(new_path)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("nii.gz"):
                full_filename = os.path.join(root, file)
                new_filename = os.path.join(new_path, "n4_corrected_" + file)
                n4_correction.n4_correction(input=full_filename, output=new_filename,
                                            n_iterations=5, n_fitting_level=5,
                                            shrink_factor=None, mask_image=None)
                assert os.path.isfile(new_filename)


def teardown_function():
    remove_tmp_dir()

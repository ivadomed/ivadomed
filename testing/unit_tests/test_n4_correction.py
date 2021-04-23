import pytest
import os
import hashlib
from ivadomed.scripts import n4_correction
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, __tmp_dir__
from testing.common_testing_util import remove_tmp_dir


def setup_function():
    create_tmp_dir()


def generate_md5(file):
    hash_md5 = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@pytest.mark.parametrize('test_lst', ['sub-unf01', 'sub-unf02', 'sub-unf03'])
def test_n4_correction(test_lst):
    files_md5 = {
        "sub-unf01_T1w.nii.gz": "7a49dad960a41644ff8d11a3dd5908cf",
        "sub-unf01_T2star.nii.gz": "fd0e85fc771bed7a6a20a19c540c0722",
        "sub-unf01_T2w.nii.gz": "9dfdf62836d1389af02ed5737e2f34ef",
        "sub-unf02_T1w.nii.gz": "b997a46434c27a89f2090be5fec7781f",
        "sub-unf02_T2star.nii.gz": "fca10610f83c1004a5a8bc29166c50d6",
        "sub-unf02_T2w.nii.gz": "5c04d7e8202ede09ea30b4cde3656021",
        "sub-unf03_T1w.nii.gz": "b658fd01e98c4c8446df153da17762a5",
        "sub-unf03_T2star.nii.gz": "51957e35ef48939f4031d9d8c3c698f9",
        "sub-unf03_T2w.nii.gz": "287056ef959d43388b6ec3173fa4ab3d"
    }
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
                md5 = generate_md5(new_filename)
                assert md5 == files_md5[file]


def teardown_function():
    remove_tmp_dir()

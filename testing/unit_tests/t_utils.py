import os
import shutil
import pytest
import nibabel as nib
import numpy as np
from ivadomed.utils import init_ivadomed
from testing.common_testing_util import remove_tmp_dir, path_repo_root, path_temp, path_data_testing_tmp, \
    path_data_testing_source, download_dataset

__test_dir__ = os.path.join(path_repo_root, 'testing/unit_tests')
__data_testing_dir__ = path_data_testing_tmp
__tmp_dir__ = path_temp

init_ivadomed()


@pytest.fixture(scope='session')
def download_data_testing_test_files():
    """
    This fixture will attempt to download test data file if there are not present.
    """
    download_dataset("data_testing")


def generate_labels():
    subdir = ['sub-unf01', 'sub-unf02', 'sub-unf03']
    files = {'T1w': ['_T1w_lesion-manual.nii.gz', '_T1w_seg-manual.nii.gz'],
             'T2star': ['_T2star_lesion-manual.nii.gz', '_T2star_seg-manual.nii.gz'],
             'T2w': ['_T2w_labels-disc-manual.nii.gz', '_T2w_lesion-manual.nii.gz',
                     '_T2w_seg-manual.nii.gz']
             }
    label_path = os.path.join(__data_testing_dir__, 'derivatives', 'labels')
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    for dir in subdir:
        source_path = os.path.join(__data_testing_dir__, dir, 'anat')
        for key in files:
            source_file = os.path.join(source_path, dir + "_" + key + ".nii.gz")
            img = nib.load(source_file)
            data = img.get_data()
            threshold = np.percentile(data, 5)
            data[data > threshold] = 0
            clipped_img = nib.Nifti1Image(data, img.affine, img.header)
            path = os.path.join(label_path, dir, "anat")
            if not os.path.exists(path):
                os.makedirs(path)
            for file in files[key]:
                file_name = os.path.join(path, dir + file)
                if not os.path.exists(file_name):
                    nib.save(clipped_img, file_name)


def create_tmp_dir(copy_data_testing_dir=True):
    """Create a temporary directory for unit_test data and copy test data files.

    1. Remove the ``tmp`` directory if it exists.
    2. Copy the ``data_testing`` directory to the ``tmp`` directory.

    Any data files created during testing will go into ``tmp`` directory.
    This is created/removed for each test.

    Args:
        copy_data_testing_dir (bool): If true, copy the __data_testing_dir_ref__ folder
            into the ``tmp`` folder.
    """
    remove_tmp_dir()
    os.mkdir(path_temp)
    if os.path.exists(path_data_testing_source) and copy_data_testing_dir:
        shutil.copytree(path_data_testing_source,
                        path_data_testing_tmp)

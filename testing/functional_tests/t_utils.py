import os
import pytest
import shutil
from ivadomed.utils import init_ivadomed
from testing.common_testing_util import remove_tmp_dir, path_repo_root, path_temp, path_data_functional_source, \
    path_data_functional_tmp, download_dataset

__test_dir__ = os.path.join(path_repo_root, 'testing/functional_tests')
__data_testing_dir__ = path_data_functional_source
__tmp_dir__ = path_temp

init_ivadomed()


@pytest.fixture(scope='session')
def get_functional_test_files():
    """
    This fixture will attempt to download test data file if there are not present.
    """
    download_dataset("data_functional_testing")


def create_tmp_dir(copy_data_testing_dir=True):
    """Create a temporary directory for data_functional and copy test data files.

    1. Remove the ``tmp`` directory if it exists.
    2. Copy the ``data_functional_testing`` directory to the ``tmp`` directory.

    Any data files created during testing will go into ``tmp`` directory.
    This is created/removed for each test.

    Args:
        copy_data_testing_dir (bool): If true, copy the __data_testing_dir_ref__ folder
            into the ``tmp`` folder.
    """
    remove_tmp_dir()
    os.mkdir(path_temp)
    if os.path.exists(path_data_functional_source) and copy_data_testing_dir:
        shutil.copytree(path_data_functional_source,
                        path_data_functional_tmp)

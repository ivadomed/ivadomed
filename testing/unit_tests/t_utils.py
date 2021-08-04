import shutil
import pytest
from pathlib import Path
from ivadomed.utils import init_ivadomed
from testing.common_testing_util import remove_tmp_dir, path_repo_root, path_temp, path_data_testing_tmp, \
    path_data_testing_source, download_dataset

__test_dir__ = Path(path_repo_root, 'testing/unit_tests')
__data_testing_dir__ = path_data_testing_tmp
__tmp_dir__ = path_temp

init_ivadomed()


@pytest.fixture(scope='session')
def download_data_testing_test_files():
    """
    This fixture will attempt to download test data file if there are not present.
    """
    download_dataset("data_testing")


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
    Path(path_temp).mkdir()
    if Path(path_data_testing_source).exists() and copy_data_testing_dir:
        shutil.copytree(path_data_testing_source,
                        path_data_testing_tmp)

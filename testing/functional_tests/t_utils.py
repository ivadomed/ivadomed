import sys
import os
import shutil
import pytest
from ivadomed.utils import init_ivadomed, __ivadomed_dir__
from ivadomed.scripts import download_data as ivadomed_download_data

__test_dir__ = os.path.join(__ivadomed_dir__, 'testing/functional_tests')
__fixtures_dir__ = os.path.join(__test_dir__, 'fixtures')
__data_testing_dir__ = "data_functional_testing"
__tmp_dir__ = "tmp"

init_ivadomed()


class bcolors(object):
    """Class for different colours."""

    normal = '\033[0m'
    red = '\033[91m'
    green = '\033[92m'
    yellow = '\033[93m'
    blue = '\033[94m'
    magenta = '\033[95m'
    cyan = '\033[96m'
    bold = '\033[1m'
    underline = '\033[4m'


def printv(string, verbose=1, type='normal'):
    """Print color-coded messages, depending on verbose status.

    Only use in command-line programs (e.g. sct_propseg).
    """
    colors = {
        'normal': bcolors.normal,
        'info': bcolors.green,
        'warning': bcolors.yellow,
        'error': bcolors.red,
        'code': bcolors.blue,
        'bold': bcolors.bold,
        'process': bcolors.magenta
    }

    if verbose:
        # The try/except is there in case stdout does not have isatty field (it did happen to me)
        try:
            # Print color only if the output is the terminal
            if sys.stdout.isatty():
                color = colors.get(type, bcolors.normal)
                print(color + string + bcolors.normal)
            else:
                print(string)
        except Exception:
            print(string)


def download_dataset(dataset='data_testing', verbose=True):
    """Download testing data from internet.

    Args:
        verbose (bool): whether or not to print
    """
    printv('\nDownloading testing data...', verbose)
    if os.path.exists(__tmp_dir__):
        __dataset_dir__ = os.path.join(__tmp_dir__, dataset)
    else:
        __dataset_dir__ = os.path.join(__test_dir__, dataset)
    ivadomed_download_data.main([
        '-d', dataset,
        '-o', __dataset_dir__
    ])


def remove_dataset(dataset='data_testing', verbose=True):
    """Recursively remove the data_testing folder.

    Args:
        verbose (bool): whether or not to print
    """
    if os.path.exists(__tmp_dir__):
        __dataset_dir__ = os.path.join(__tmp_dir__, dataset)
    else:
        __dataset_dir__ = os.path.join(__test_dir__, dataset)
    printv("rm -rf %s" % (__dataset_dir__), verbose=verbose, type="code")
    shutil.rmtree(__dataset_dir__, ignore_errors=True)


def create_tmp_dir(copy_data_testing_dir=True):
    """Create temporary directory for test data and copy test data files.

    1. Remove the ``tmp`` directory if it exists.
    2. Copy the ``data_functional_testing`` directory to the ``tmp`` directory.

    Any data files created during testing will go into ``tmp`` directory.
    This is created/removed for each test.

    Args:
        copy_data_testing_dir (bool): If true, copy the __data_testing_dir_ref__ folder
            into the ``tmp`` folder.
    """
    remove_tmp_dir()
    os.mkdir(__tmp_dir__)
    if os.path.exists(__data_testing_dir__) and copy_data_testing_dir:
        shutil.copytree(__data_testing_dir__,
                        os.path.join(__tmp_dir__, __data_testing_dir__))


def remove_tmp_dir():
    """Recursively remove the ``tmp`` directory if it exists."""
    shutil.rmtree(__tmp_dir__, ignore_errors=True)


@pytest.fixture(scope='session')
def get_functional_test_files():
    """
    This fixture will attempt to download test data file if there are not present.
    """
    download_dataset(__data_testing_dir__)

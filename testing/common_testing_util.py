from pathlib import Path
from loguru import logger
import pytest
from ivadomed.scripts import download_data as ivadomed_download_data
import shutil
import sys

# Cannot use __ivadomed_dir__ here as that would refer to the IvadoMed installed in the Conda/Venv
# Absolute path to the temporary folder, which should sit at the root of the repo.
# We could also replace this hard coded levels reference with a simple loop to search for .git/.github sitting at the
# root of the repo. To be discussed.
FOLDER_LEVELS_FROM_ROOT = 1

path_repo_root: Path = Path(__file__).parents[FOLDER_LEVELS_FROM_ROOT].absolute()

path_temp: str = str(path_repo_root / "tmp")

path_unit_tests: str = str(path_repo_root / "testing" / "unit_tests")
path_data_testing_source: str = str(path_repo_root / "data_testing")
path_data_testing_tmp: str = str(path_repo_root / "tmp" / Path(path_data_testing_source).name)

path_functional_tests: str = str(path_repo_root / "testing" / "functional_tests")
path_data_functional_source: str = str(path_repo_root / "data_functional_testing")
path_data_functional_tmp: str = str(path_repo_root / "tmp" / Path(path_data_functional_source).name)


def download_dataset(dataset: str = 'data_testing'):
    """Download testing data from internet.

    Args:
        dataset (str): the name of the dataset to download
    """

    path_dataset: Path = path_repo_root / dataset

    # Early abort if testing data already exist.
    if path_dataset.exists():
        logger.info(f"\nTesting data files appear to already exist at {path_dataset}, aborting download")
        return

    logger.info(f'\nDownloading testing data... to {dataset}')

    # Call the ivadomed download CLI
    ivadomed_download_data.main([
        '-d', dataset,
        '-o', str(path_dataset)
    ])


def remove_dataset(dataset: str = 'data_testing'):
    """Recursively remove the data_testing folder.

    Args:
        dataset (str): the name of the dataset to remove
    """

    path_dataset = Path(path_temp, dataset)

    logger.debug("rm -rf %s" % path_dataset)

    shutil.rmtree(path_dataset, ignore_errors=True)


def remove_tmp_dir():
    """Recursively remove the ``tmp`` directory if it exists."""
    shutil.rmtree(path_temp, ignore_errors=True)


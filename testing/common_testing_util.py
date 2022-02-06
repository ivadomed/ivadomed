import os
from pathlib import Path

import json
import csv_diff
import pandas as pd
import pytest
from loguru import logger

from ivadomed.keywords import BidsDataFrameKW, LoaderParamsKW, ContrastParamsKW, ConfigKW
from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed.loader import loader as imed_loader
from ivadomed.scripts import download_data as ivadomed_download_data
from ivadomed.main import set_loader_params
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

path_data_multi_sessions_contrasts_source: Path = path_repo_root / "data_multi_testing"
path_data_multi_sessions_contrasts_tmp: Path = path_repo_root / "tmp" / Path(
    path_data_multi_sessions_contrasts_source).name


def download_dataset(dataset: str = 'data_testing', verbose=True):
    """Download testing data from internet.

    Args:
        dataset (str): the name of the dataset to download
        verbose (bool): whether or not to print
    """

    path_dataset: Path = path_repo_root / dataset

    # Early abort if testing data already exist.
    if path_dataset.exists():
        printv(f'\nTesting data files appear to already exist at {path_dataset}, aborting download', verbose)
        return

    printv(f'\nDownloading testing data... to {dataset}', verbose)

    # Call the ivadomed download CLI
    ivadomed_download_data.main([
        '-d', dataset,
        '-o', str(path_dataset)
    ])


def remove_dataset(dataset: str = 'data_testing', verbose=True):
    """Recursively remove the data_testing folder.

    Args:
        dataset (str): the name of the dataset to remove
        verbose (bool): whether or not to print
    """

    path_dataset = Path(path_temp, dataset)

    printv("rm -rf %s" % (path_dataset), verbose=verbose, type="code")

    shutil.rmtree(path_dataset, ignore_errors=True)


def remove_tmp_dir():
    """Recursively remove the ``tmp`` directory if it exists."""
    shutil.rmtree(path_temp, ignore_errors=True)


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


def assert_empty_bids_dataframe(loader_parameters: dict):
    """Assertion function used during unit test to check if the loader parameters result in an empty bids dataframes.
    Used as part of pytest-case scenarios for when the Step1/2/3 filters excluded almost all data. (e.g. when no
    participants have the required Session data)

    Args:
        loader_parameters (dict): a dictionary object containing all loader parameters necessary derived from
        config.json to construct the BidsDataFrame
    """

    # Create the bids frame.
    bids_df = BidsDataframe(loader_parameters,
                            str(path_data_multi_sessions_contrasts_tmp),
                            derivatives=True)

    # Assert new dataframe EQUALs an empty pandas dataframe
    assert bids_df.df.equals(pd.DataFrame())


def bids_dataframe_comparison_framework(loader_parameters: dict,
                                        path_output: str,
                                        target_csv: str,
                                        test_csv: str = "df_test.csv"):
    """Main test function used to set up a CSV comparison between expected vs the output from the test
    Args:
        loader_parameters (dict): a dictionary object containing all loader parameters necessary derived from
        config.json to construct the BidsDataFrame
        path_output: Output folder
        target_csv (str): the filename string  of a CSV which indicate the goal of the unit test scenario. It is
        derived from the bids_dataframe after stripping out its PATH column.
        test_csv (str): the filename string of a CSV which is produced from Bids Dataframe
    """
    # Create the bids frame.
    bids_df = BidsDataframe(loader_parameters,
                            path_output,
                            derivatives=True)

    # Drop path as that can varies across runs.
    df_test = bids_df.df.drop(columns=[BidsDataFrameKW.PATH])

    # Sorting to ensure consistencies when comparing with ground truth.
    df_test = df_test.sort_values(by=[BidsDataFrameKW.FILENAME]).reset_index(drop=True)

    # Generate full path to the target reference that we hope to match (e.g. df_ref.csv or any other scenario specific
    # CSV file name
    csv_ref = os.path.join(loader_parameters[LoaderParamsKW.PATH_DATA][0], target_csv)

    # Generate full path for the csv which is produced from Bids Dataframe so that we can save the CSV to that location
    csv_test = os.path.join(loader_parameters[LoaderParamsKW.PATH_DATA][0], test_csv)
    df_test.to_csv(csv_test, index=False)

    # Calculate differences between the generated versus the ground truth by comparing the output csv (csv_test) with
    # the target reference CSV from the data rep (csv_ref)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))

    # Any modification is in relation to csv_test - csv_ref direction. i.e. Add = what did the csv_diff Add in
    # relation to the csv_ref.
    assert diff == {'added': [], 'removed': [], 'changed': [], 'columns_added': [], 'columns_removed': []}


def get_multi_default_case() -> dict:
    """Generate a default case loader json for the multi-session configuration JSON by:
    1) load the default config.json
    2) load its loader parameters
    3) overwrite its various information in relation to the data_multi unit test data most notably for
        a) target sessions
        b) path data
        c) target suffix
        d) contrast parameters

    returns:  the unit test ready loader parameters
    """

    # Load the default config.json from config folder.
    path_config_json = Path(__file__).parent.parent / "ivadomed" / "config" / "config.json"
    with path_config_json.open("r") as config:
        dict_config: dict = json.load(config)

    # Update its loader parameters as though we are preparing for training and get its loader parameter
    default_loader_parameters = set_loader_params(dict_config, is_train=True)

    # A default dict which subsequent tests attempt to deviate from
    default_loader_parameters.update({
        LoaderParamsKW.MULTICHANNEL: "true",
        LoaderParamsKW.TARGET_SESSIONS: ["01", "02", "03", "04"],
        LoaderParamsKW.PATH_DATA: [str(path_data_multi_sessions_contrasts_tmp)],
        LoaderParamsKW.TARGET_SUFFIX: ["_lesion-manual-rater1", "_lesion-manual-rater2"],
        LoaderParamsKW.EXTENSIONS: [".nii", ".nii.gz"],
        LoaderParamsKW.ROI_PARAMS: {"suffix": None, "slice_filter_roi": None},
        LoaderParamsKW.CONTRAST_PARAMS: {
            ContrastParamsKW.CONTRAST_LIST: ["T1w", "T2w", "FLAIR", "PD"]
        }
    })

    return default_loader_parameters


def prepare_loader_dataset(loader_parameters, model_parameters, transform_parameters, path_output):
    """
    Loader the dataset for test in 2d patching,2d png and target_suffix
    """

    loader_parameters.update({"model_params": model_parameters})
    bids_df = BidsDataframe(loader_parameters, path_output, derivatives=True)
    data_lst = ['sub-rat3_ses-01_sample-data9_SEM.png']
    ds = imed_loader.load_dataset(bids_df,
                                  **{**loader_parameters, **{'data_list': data_lst,
                                                             'transforms_params': transform_parameters,
                                                             'dataset_type': 'training'}})

    return ds

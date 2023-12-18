import os
import json
from pathlib import Path

from ivadomed.keywords import ConfigKW, LoaderParamsKW
from testing.functional_tests.t_utils import create_tmp_dir, __data_testing_dir__, download_functional_test_files, \
    __tmp_dir__
from testing.common_testing_util import remove_tmp_dir
from testing.mocker.mocker_fixture import create_mock_bids_file_structures, create_example_mock_bids_file_structures
from testing.common_testing_util import path_temp
from typing import Tuple
from loguru import logger
import pytest
from ivadomed.main import run_command
from ivadomed.config.example_loader_v2_configs import example_2i1o_all_dataset_groups_config_json, \
    example_1i1o_all_dataset_groups_config_json, path_mock_data
import ivadomed.config as config

def setup_function():
    create_tmp_dir(copy_data_testing_dir=False)

@pytest.mark.parametrize('input_file_dataset', [
    example_2i1o_all_dataset_groups_config_json,
    example_1i1o_all_dataset_groups_config_json,
])
def test_training_with_filedataset(
        download_functional_test_files,  # pytest fixture, do not remove.
        input_file_dataset
):
    create_example_mock_bids_file_structures(path_mock_data),  # pytest fixture, do not remove.

    json_data, _ = patch_existing_json_for_test(input_file_dataset)

    # Call ivado cmd_train
    best_training_dice, best_training_loss, best_validation_dice, best_validation_loss = run_command(context=json_data)


def patch_existing_json_for_test(
        input_file_dataset: dict,
        json_specification: str = "automate_training_config",
        save_patched_to_file:bool = False) -> Tuple[dict, str]:
    """
    Common code to patch an existing Laoder V1 JSON to bring it to Loader V2 standard
    Args:
        input_file_dataset:

    Returns:

    """
    # Build the config file
    path_default_config = str(Path(__data_testing_dir__, f'{json_specification}.json'))
    with open(path_default_config) as json_file:
        json_data: dict = json.load(json_file)

    # Add the new key to JSON.
    json_data.update(input_file_dataset)

    # Popping out the contract key to enable it to AUTO using the new LoaderConfiguration
    json_data[ConfigKW.LOADER_PARAMETERS].pop(LoaderParamsKW.CONTRAST_PARAMS)

    # Pop in the data path to enable data versioning
    json_data[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.PATH_DATA] = path_temp

    # Patching in the two required parameters.
    json_data["path_output"] = "pytest_output_folder"
    json_data["log_file"] = "log"
    # Patching old automate_training_config.json to use the new balance_samples
    json_data["training_parameters"]["balance_samples"] = {
        "applied": False,
        "type": "gt"}
    # Patch default model to
    json_data["default_model"]["is_2d"] = True
    # Debug print out JSON
    logger.trace(json.dumps(json_data, indent=4))

    path_default_config_patched = ""

    if save_patched_to_file:
        # Build the config file
        path_default_config_patched = os.path.join(__data_testing_dir__, f'{json_specification}_patched.json')

        # Write out the patched existing JSON to file to allow CLI access
        with open(path_default_config_patched, "w") as json_file:
            json.dump(json_data, json_file)

    return json_data, path_default_config_patched


@pytest.mark.script_launch_mode('subprocess')
@pytest.mark.parametrize('input_file_dataset', [
    example_2i1o_all_dataset_groups_config_json,
    example_1i1o_all_dataset_groups_config_json,
])
def test_training_cli(
        download_functional_test_files,  # pytest fixture, do not remove.
        input_file_dataset,
        script_runner
):
    create_example_mock_bids_file_structures(path_mock_data),  # pytest fixture, do not remove.

    json_data, path_default_config_patched = patch_existing_json_for_test(input_file_dataset, save_patched_to_file=True)

    __output_dir__ = Path(__tmp_dir__, 'results')

    ret = script_runner.run('ivadomed', '--train', '-c', f'{path_default_config_patched}',
                            '--path-data', f'{__data_testing_dir__}',
                            '-po', f'{__output_dir__}')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")

    assert ret.success


@pytest.mark.skip(reason="To be Implemented")
def test_training_with_bidsdataset(download_functional_test_files):
    pass


@pytest.mark.skip(reason="To be Implemented")
def test_training_with_regex_dataset(download_functional_test_files):
    pass


@pytest.mark.skip(reason="To be Implemented")
def test_training_with_consolidated_dataset(download_functional_test_files):
    pass


def teardown_function():
     remove_tmp_dir()

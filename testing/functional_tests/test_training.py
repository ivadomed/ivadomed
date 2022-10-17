import os
import json

from ivadomed.keywords import ConfigKW, LoaderParamsKW
from testing.functional_tests.t_utils import create_tmp_dir, __data_testing_dir__, download_functional_test_files
from testing.common_testing_util import remove_tmp_dir

from loguru import logger
import pytest
from ivadomed.main import run_command
from ivadomed.config.example_loader_v2_configs import example_all_dataset_groups_config_json, example_uni_channel_all_dataset_groups_config_json

def setup_function():
    create_tmp_dir()

@pytest.mark.parametrize('input_file_dataset', [
    # example_all_dataset_groups_config_json,
    example_uni_channel_all_dataset_groups_config_json,
])
def test_training_with_filedataset(download_functional_test_files, input_file_dataset):


    # Build the config file
    path_default_config = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    with open(path_default_config) as json_file:
        json_data: dict = json.load(json_file)

    # Add the new key to JSON.
    # json_data.update(example_uni_channel_all_dataset_groups_config_json)
    # Update three major keys.

    # update 1) dataset_groups, 2) expected_input, 3) expected_gt
    json_data.update(example_all_dataset_groups_config_json)

    # Popping out the contract key to enable it to AUTO using the new LoaderConfiguration
    json_data[ConfigKW.LOADER_PARAMETERS].pop(LoaderParamsKW.CONTRAST_PARAMS)

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

    # Build loader parameter?

    # Build the model parameters
    # Build the Generalized Loader Configuration

    # Build the example dataset

    # Call ivado cmd_train
    best_training_dice, best_training_loss, best_validation_dice, best_validation_loss = run_command(context=json_data)


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

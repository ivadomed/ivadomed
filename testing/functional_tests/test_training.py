import os
import json

from ivadomed.keywords import ConfigKW, LoaderParamsKW
from testing.functional_tests.t_utils import create_tmp_dir, __data_testing_dir__, \
    download_functional_test_files
from testing.common_testing_util import remove_tmp_dir

from loguru import logger
import pytest
from ivadomed.main import run_command

def setup_function():
    create_tmp_dir()


example_uni_channel_all_dataset_groups_config_json: dict = {
    "dataset_groups": [
        {
            "dataset_group_label": "DataSetGroup1",
            "training": [
                {
                    "type": "FILES",
                    "subset_label": "TrainFileDataSet1",
                    "image_ground_truth": [
                        [["sub-01/ses-01/anat/sub-01_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-01/ses-01/anat/sub-01_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-01/ses-01/anat/sub-01_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-01/ses-02/anat/sub-01_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-01/ses-02/anat/sub-01_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-01/ses-02/anat/sub-01_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-01/ses-03/anat/sub-01_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-01/ses-03/anat/sub-01_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-01/ses-03/anat/sub-01_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-01/ses-03/anat/sub-01_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-01/ses-04/anat/sub-01_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-01/ses-04/anat/sub-01_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 1,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",
                },
            ],
            "validation": [
                {
                    "type": "FILES",
                    "subset_label": "ValFileDataSet1",
                    "image_ground_truth": [
                        [["sub-02/ses-01/anat/sub-02_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-02/ses-01/anat/sub-02_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-02/ses-01/anat/sub-02_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-02/ses-02/anat/sub-02_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-02/ses-02/anat/sub-02_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-02/ses-02/anat/sub-02_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-02/ses-03/anat/sub-02_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-02/ses-03/anat/sub-02_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-02/ses-03/anat/sub-02_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-02/ses-03/anat/sub-02_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-02/ses-04/anat/sub-02_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-02/ses-04/anat/sub-02_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 1,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",

                }
            ],
            "test": [
                {
                    "type": "FILES",
                    "subset_label": "TestFileDataSet1",
                    "image_ground_truth": [
                        [["sub-03/ses-01/anat/sub-03_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-03/ses-01/anat/sub-03_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-03/ses-01/anat/sub-03_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-03/ses-02/anat/sub-03_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-03/ses-02/anat/sub-03_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-03/ses-02/anat/sub-03_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-03/ses-03/anat/sub-03_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-03/ses-03/anat/sub-03_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-03/ses-03/anat/sub-03_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-03/ses-03/anat/sub-03_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-03/ses-04/anat/sub-03_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-03/ses-04/anat/sub-03_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 1,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",
                }
            ],

        },
        {
            "dataset_group_label": "DataSetGroup2",
            "training": [
                {
                    "type": "FILES",
                    "subset_label": "TrainFileDataSet1",
                    "image_ground_truth": [
                        [["sub-04/ses-01/anat/sub-04_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-04/ses-01/anat/sub-04_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-04/ses-01/anat/sub-04_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-04/ses-02/anat/sub-04_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-04/ses-02/anat/sub-04_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-04/ses-02/anat/sub-04_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-04/ses-03/anat/sub-04_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-04/ses-03/anat/sub-04_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-04/ses-03/anat/sub-04_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-04/ses-03/anat/sub-04_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-04/ses-04/anat/sub-04_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-04/ses-04/anat/sub-04_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 1,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",
                },
            ],
            "validation": [
                {
                    "type": "FILES",
                    "subset_label": "ValFileDataSet1",
                    "image_ground_truth": [
                        [["sub-05/ses-01/anat/sub-05_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-05/ses-01/anat/sub-05_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-05/ses-01/anat/sub-05_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-05/ses-02/anat/sub-05_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-05/ses-02/anat/sub-05_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-05/ses-02/anat/sub-05_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-05/ses-03/anat/sub-05_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-05/ses-03/anat/sub-05_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-05/ses-03/anat/sub-05_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-05/ses-03/anat/sub-05_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-05/ses-04/anat/sub-05_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-05/ses-04/anat/sub-05_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 1,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",
                }
            ],
            "test": [
                {
                    "type": "FILES",
                    "subset_label": "TestFileDataSet1",
                    "image_ground_truth": [
                        [["sub-06/ses-01/anat/sub-06_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-06/ses-01/anat/sub-06_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-06/ses-01/anat/sub-06_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-06/ses-02/anat/sub-06_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-06/ses-02/anat/sub-06_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-06/ses-02/anat/sub-06_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-06/ses-03/anat/sub-06_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-06/ses-03/anat/sub-06_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-06/ses-03/anat/sub-06_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-06/ses-03/anat/sub-06_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-06/ses-04/anat/sub-06_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-06/ses-04/anat/sub-06_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 1,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",
                }
            ],

        },
    ]
}


def test_training_with_filedataset(download_functional_test_files):
    from ivadomed.loader.all_dataset_group import example_all_dataset_groups_config_json

    # Build the config file
    path_default_config = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    with open(path_default_config) as json_file:
        json_data: dict = json.load(json_file)

    # Add the new key to JSON.
    # json_data.update(example_uni_channel_all_dataset_groups_config_json)
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

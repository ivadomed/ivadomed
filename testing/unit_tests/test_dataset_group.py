from ivadomed.keywords import DataloaderKW
from ivadomed.loader.dataset_group import DatasetGroup
from ivadomed.loader.files_dataset import FilesDataset
from testing.common_testing_util import remove_tmp_dir
from testing.mocker.mocker_fixture import create_mock_bids_file_structures
from testing.unit_tests.t_utils import (
    create_tmp_dir
)
from ivadomed.config.example_loader_v2_configs import example_2i1o_all_dataset_groups_config_json
from ivadomed.loader.generalized_loader_configuration import (
    GeneralizedLoaderConfiguration,
)
from ivadomed.config.example_loader_v2_configs import example_2i1o_all_dataset_groups_config_json, \
    example_1i1o_all_dataset_groups_config_json, path_mock_data

def setup_function():
    create_tmp_dir()

def test_dataset_group():

    create_mock_bids_file_structures(path_mock_data),  # pytest fixture, do not remove.

    model_dict = {
        "name": "Unet",
        "dropout_rate": 0.3,
        "bn_momentum": 0.1,
        "final_activation": "sigmoid",
        "depth": 3,
        "is_2d": True
    }

    # Build a GeneralizedLoaderConfiguration:
    loader_config: GeneralizedLoaderConfiguration = GeneralizedLoaderConfiguration(
        model_params=model_dict,
    )

    # Get the mock JSON which has 2 inputs and 1 output, get their first DatasetGroup
    example_DatasetGroup_json = example_2i1o_all_dataset_groups_config_json.get(DataloaderKW.DATASET_GROUPS)[0]

    a = DatasetGroup(example_DatasetGroup_json, loader_config)
    a.preview(verbose=True)


def teardown_function():
    remove_tmp_dir()

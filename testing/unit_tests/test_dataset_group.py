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

    a_dataset_group = DatasetGroup(example_DatasetGroup_json, loader_config)
    a_dataset_group.preview(verbose=True)

    # Should have 3 files pairs, each of them should have 2 Input and 1 Output
    for data in [a_dataset_group.train_filename_pairs, a_dataset_group.val_filename_pairs, a_dataset_group.test_filename_pairs]:
        assert len(data) == 3
        for pair in data:
            assert len(pair[0]) == 2
            assert len(pair[1]) == 1


def teardown_function():
    remove_tmp_dir()

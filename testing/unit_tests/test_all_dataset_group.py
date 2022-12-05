from ivadomed.loader.all_dataset_group import AllDatasetGroups
from ivadomed.config.example_loader_v2_configs import example_2i1o_all_dataset_groups_config_json, path_mock_data
from ivadomed.loader.dataset_group import FileDatasetGroup

from testing.common_testing_util import remove_tmp_dir
from testing.mocker.mocker_fixture import create_mock_bids_file_structures, create_example_mock_bids_file_structures
from testing.unit_tests.t_utils import (
    create_tmp_dir
)
from ivadomed.loader.generalized_loader_configuration import (
    GeneralizedLoaderConfiguration,
)


def setup_function():
    create_tmp_dir()


def test_all_dataset_group():

    create_example_mock_bids_file_structures(path_mock_data),  # pytest fixture, do not remove.

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
    all_dataset_groups = AllDatasetGroups(example_2i1o_all_dataset_groups_config_json, loader_config)
    all_dataset_groups.preview(verbose=True)

    for a_dataset_group in all_dataset_groups.list_dataset_groups:
        # Should have 3 files pairs, each of them should have 2 Input and 1 Output
        for data in [a_dataset_group.train_filename_pairs, a_dataset_group.val_filename_pairs, a_dataset_group.test_filename_pairs]:
            assert len(data) == 3
            for pair in data:
                assert len(pair[0]) == 2
                assert len(pair[1]) == 1

def teardown_function():
    remove_tmp_dir()

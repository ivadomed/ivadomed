from ivadomed.config.example_loader_v2_configs import example_FileDataset_json, path_mock_data
from ivadomed.loader.files_dataset import FilesDataset
from testing.common_testing_util import remove_tmp_dir
from testing.mocker.mocker_fixture import create_mock_bids_file_structures
from testing.unit_tests.t_utils import (
    create_tmp_dir
)
from ivadomed.loader.generalized_loader_configuration import (
    GeneralizedLoaderConfiguration,
)


def setup_function():
    create_tmp_dir()

def test_FilesDataset():

    create_mock_bids_file_structures(path_mock_data),  # pytest fixture, do not remove.

    model_dict = {
        "name": "Unet",
        "dropout_rate": 0.3,
        "bn_momentum": 0.1,
        "final_activation": "sigmoid",
        "depth": 3,
        "is_2d": True # key params that must exist.
    }

    # Build a GeneralizedLoaderConfiguration:
    model_config_json: GeneralizedLoaderConfiguration = GeneralizedLoaderConfiguration(
        model_params=model_dict,
    )
    a = FilesDataset(example_FileDataset_json, model_config_json)
    a.preview(verbose=True)
    print(len(a))

def teardown_function():
    remove_tmp_dir()
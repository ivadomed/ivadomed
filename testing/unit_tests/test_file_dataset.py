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
from ivadomed.keywords import ModelParamsKW
import pytest

def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('model_dict', [{
    ModelParamsKW.NAME: "Unet",
    ModelParamsKW.DROPOUT_RATE: 0.3,
    "bn_momentum": 0.1,
    "final_activation": "sigmoid",
    "depth": 3,
    "is_2d": True  # key params that must exist.
}])
def test_FilesDataset(model_dict: dict):

    create_mock_bids_file_structures(path_mock_data),  # pytest fixture, do not remove.
    # Build a GeneralizedLoaderConfiguration:
    model_config_json: GeneralizedLoaderConfiguration = GeneralizedLoaderConfiguration(
        model_params=model_dict,
    )
    a = FilesDataset(example_FileDataset_json, model_config_json)
    a.preview(verbose=True)

    # Should have 4 files pairs, each of them should have 2 Input and 1 Output
    assert len(a.filename_pairs) == 4
    for pair in a.filename_pairs:
        assert len(pair[0]) == 2
        assert len(pair[1]) == 1

def teardown_function():
    remove_tmp_dir()

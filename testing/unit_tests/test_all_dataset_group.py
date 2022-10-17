from ivadomed.loader.all_dataset_group import AllDatasetGroups
from ivadomed.config.example_loader_v2_configs import example_2i1o_all_dataset_groups_config_json
from ivadomed.loader.dataset_group import DatasetGroup

from testing.common_testing_util import remove_tmp_dir
from testing.unit_tests.t_utils import (
    create_tmp_dir
)
from ivadomed.loader.generalized_loader_configuration import (
    GeneralizedLoaderConfiguration,
)


def setup_function():
    create_tmp_dir()


def test_all_dataset_group():
    model_dict = {
        "name": "Unet",
        "dropout_rate": 0.3,
        "bn_momentum": 0.1,
        "final_activation": "sigmoid",
        "depth": 3,
    }

    # Build a GeneralizedLoaderConfiguration:
    loader_config: GeneralizedLoaderConfiguration = GeneralizedLoaderConfiguration(
        model_params=model_dict,
    )
    a = AllDatasetGroups(example_2i1o_all_dataset_groups_config_json, loader_config)
    a.preview(verbose=True)


def teardown_function():
    remove_tmp_dir()

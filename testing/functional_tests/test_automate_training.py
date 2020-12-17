import logging
import os
from cli_base import remove_tmp_dir, __tmp_dir__, download_dataset, remove_dataset, create_tmp_dir
from ivadomed.scripts import automate_training
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()
    download_dataset("data_example_spinegeneric")


def test_automate_training():
    config_file = os.path.join(__tmp_dir__, 'automate_training_config_test.json')
    param_file = os.path.join(__tmp_dir__,
                              'automate_training_hyperparameter_opt.json')
    automate_training.main(args=[
        '--config', f'{config_file}',
        '--params', f'{param_file}'
    ])


def teardown_function():
    remove_dataset("data_example_spinegeneric")
    remove_tmp_dir()

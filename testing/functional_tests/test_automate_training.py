import logging
import os
from functional_tests.t_utils import remove_tmp_dir, __tmp_dir__, create_tmp_dir, __data_testing_dir__
from ivadomed.scripts import automate_training
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_automate_training():
    config_file = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    param_file = os.path.join(__data_testing_dir__,
                              'automate_training_hyperparameter_opt.json')
    __output_dir__ = os.path.join(__tmp_dir__, 'results')
    automate_training.main(args=[
        '--config', f'{config_file}',
        '--params', f'{param_file}',
        '--output_dir', f'{__output_dir__}'
    ])
    assert os.path.exists(os.path.join(__output_dir__, 'detailed_results.csv'))
    assert os.path.exists(os.path.join(__output_dir__, 'temporary_results.csv'))
    assert os.path.exists(os.path.join(__output_dir__, 'average_eval.csv'))


def teardown_function():
    remove_tmp_dir()

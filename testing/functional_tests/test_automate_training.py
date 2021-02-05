import logging
import os
import pytest
from functional_tests.t_utils import remove_tmp_dir, __tmp_dir__, create_tmp_dir, __data_testing_dir__
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


@pytest.mark.script_launch_mode('subprocess')
def test_automate_training(script_runner):
    config_file = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    param_file = os.path.join(__data_testing_dir__,
                              'automate_training_hyperparameter_opt.json')
    __output_dir__ = os.path.join(__tmp_dir__, 'results')

    ret = script_runner.run('ivadomed_automate_training', '--config', f'{config_file}',
                            '--params', f'{param_file}',
                            '--path-data', f'{__data_testing_dir__}',
                            '--output_dir', f'{__output_dir__}')
    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success
    assert os.path.exists(os.path.join(__output_dir__, 'detailed_results.csv'))
    assert os.path.exists(os.path.join(__output_dir__, 'temporary_results.csv'))
    assert os.path.exists(os.path.join(__output_dir__, 'average_eval.csv'))


@pytest.mark.script_launch_mode('subprocess')
def test_automate_training_run_test(script_runner):
    config_file = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    param_file = os.path.join(__data_testing_dir__,
                              'automate_training_hyperparameter_opt.json')
    __output_dir__ = os.path.join(__tmp_dir__, 'results')

    ret = script_runner.run('ivadomed_automate_training', '--config', f'{config_file}',
                            '--params', f'{param_file}',
                            '--path-data', f'{__data_testing_dir__}',
                            '--output_dir', f'{__output_dir__}',
                            '--run-test')
    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success
    assert os.path.exists(os.path.join(__output_dir__, 'detailed_results.csv'))
    assert os.path.exists(os.path.join(__output_dir__, 'temporary_results.csv'))
    assert os.path.exists(os.path.join(__output_dir__, 'average_eval.csv'))


def teardown_function():
    remove_tmp_dir()

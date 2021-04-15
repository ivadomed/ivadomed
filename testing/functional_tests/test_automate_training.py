import logging
import os
import pytest
from testing.functional_tests.t_utils import __tmp_dir__, create_tmp_dir, __data_testing_dir__, download_functional_test_files
from testing.common_testing_util import remove_tmp_dir
from ivadomed import config_manager as imed_config_manager
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


@pytest.mark.script_launch_mode('subprocess')
def test_automate_training(download_functional_test_files, script_runner):
    file_config = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    file_config_hyper = os.path.join(__data_testing_dir__,
                                     'automate_training_hyperparameter_opt.json')
    __output_dir__ = os.path.join(__tmp_dir__, 'results')

    ret = script_runner.run('ivadomed_automate_training', '--config', f'{file_config}',
                            '--config-hyper', f'{file_config_hyper}',
                            '--path-data', f'{__data_testing_dir__}',
                            '--output_dir', f'{__output_dir__}')
    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success
    assert os.path.exists(os.path.join(__output_dir__, 'detailed_results.csv'))
    assert os.path.exists(os.path.join(__output_dir__, 'temporary_results.csv'))
    assert os.path.exists(os.path.join(__output_dir__, 'average_eval.csv'))

    # check sha256 is recorded in config_file.json
    initial_config = imed_config_manager.ConfigurationManager(file_config).get_config()
    result = []
    name = "config_file.json"
    for root, dirs, files in os.walk(os.path.dirname(initial_config["path_output"])):
        if name in files:
            result.append(os.path.join(root, name))
    assert result != []
    for generated_config in result:
        config = imed_config_manager.ConfigurationManager(generated_config).get_config()
        assert 'training_sha256' in config


@pytest.mark.script_launch_mode('subprocess')
def test_automate_training_run_test(download_functional_test_files, script_runner):
    file_config = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    file_config_hyper = os.path.join(__data_testing_dir__,
                                     'automate_training_hyperparameter_opt.json')
    __output_dir__ = os.path.join(__tmp_dir__, 'results')

    ret = script_runner.run('ivadomed_automate_training', '--config', f'{file_config}',
                            '--config-hyper', f'{file_config_hyper}',
                            '--path-data', f'{__data_testing_dir__}',
                            '--output_dir', f'{__output_dir__}',
                            '--run-test')
    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success
    assert os.path.exists(os.path.join(__output_dir__, 'detailed_results.csv'))
    assert os.path.exists(os.path.join(__output_dir__, 'temporary_results.csv'))
    assert os.path.exists(os.path.join(__output_dir__, 'average_eval.csv'))

    # check sha256 is recorded in config_file.json
    initial_config = imed_config_manager.ConfigurationManager(file_config).get_config()
    result = []
    name = "config_file.json"
    for root, dirs, files in os.walk(os.path.dirname(initial_config["path_output"])):
        if name in files:
            result.append(os.path.join(root, name))
    assert result != []
    for generated_config in result:
        config = imed_config_manager.ConfigurationManager(generated_config).get_config()
        assert 'training_sha256' in config


def teardown_function():
    remove_tmp_dir()

import logging
import pytest
from pytest_console_scripts import script_runner
from pathlib import Path
from testing.functional_tests.t_utils import __tmp_dir__, create_tmp_dir, __data_testing_dir__, \
    download_functional_test_files, check_sha256
from testing.common_testing_util import remove_tmp_dir

logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


@pytest.mark.script_launch_mode('subprocess')
def test_automate_training(download_functional_test_files, script_runner):
    file_config = Path(__data_testing_dir__, 'automate_training_config.json')
    file_config_hyper = Path(__data_testing_dir__, 'automate_training_hyperparameter_opt.json')
    __output_dir__ = Path(__tmp_dir__, 'results')

    ret = script_runner.run('ivadomed_automate_training', '--config', f'{file_config}',
                            '--config-hyper', f'{file_config_hyper}',
                            '--path-data', f'{__data_testing_dir__}',
                            '--output_dir', f'{__output_dir__}')
    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success
    assert Path(__output_dir__, 'detailed_results.csv').exists()
    assert Path(__output_dir__, 'temporary_results.csv').exists()
    assert Path(__output_dir__, 'average_eval.csv').exists()

    # check sha256 is recorded in config_file.json
    check_sha256(str(file_config))


@pytest.mark.script_launch_mode('subprocess')
def test_automate_training_run_test(download_functional_test_files, script_runner):
    file_config = Path(__data_testing_dir__, 'automate_training_config.json')
    file_config_hyper = Path(__data_testing_dir__, 'automate_training_hyperparameter_opt.json')
    __output_dir__ = Path(__tmp_dir__, 'results')

    ret = script_runner.run('ivadomed_automate_training', '--config', f'{file_config}',
                            '--config-hyper', f'{file_config_hyper}',
                            '--path-data', f'{__data_testing_dir__}',
                            '--output_dir', f'{__output_dir__}',
                            '--run-test')
    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success
    assert Path(__output_dir__, 'detailed_results.csv').exists()
    assert Path(__output_dir__, 'temporary_results.csv').exists()
    assert Path(__output_dir__, 'average_eval.csv').exists()

    # check sha256 is recorded in config_file.json
    check_sha256(str(file_config))


# def teardown_function():
#     remove_tmp_dir()

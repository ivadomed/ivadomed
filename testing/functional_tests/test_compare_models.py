import logging
import os
from cli_base import remove_tmp_dir, __tmp_dir__, create_tmp_dir, download_dataset
from ivadomed.scripts import compare_models
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()
    download_dataset("data_functional_testing")


def test_compare_models():
    __output_file__ = os.path.join(__tmp_dir__, 'comparison_results.csv')
    path_df = os.path.join(__tmp_dir__, 'data_functional_testing', 'temporary_results.csv')
    compare_models.main(args=['-df', path_df,
                              '-n', '2',
                              '-o', __output_file__])
    assert os.path.exists(__output_file__)


def teardown_function():
    remove_tmp_dir()

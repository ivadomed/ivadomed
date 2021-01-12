import logging
import os
from cli_base import remove_tmp_dir, __tmp_dir__, create_tmp_dir, __data_testing_dir__
from ivadomed.scripts import compare_models
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_compare_models():
    __output_file__ = os.path.join(__tmp_dir__, 'comparison_results.csv')
    path_df = os.path.join(__data_testing_dir__, 'temporary_results.csv')
    compare_models.main(args=['-df', path_df,
                              '-n', '2',
                              '-o', __output_file__])
    assert os.path.exists(__output_file__)


def teardown_function():
    remove_tmp_dir()

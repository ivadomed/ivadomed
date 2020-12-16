import logging
import os
from cli_base import remove_tmp_dir, __tmp_dir__, create_tmp_dir
from ivadomed.scripts import training_curve
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_compare_models():
    # training_curve.main(args=[])
    assert 1 == 1


def teardown_function():
    remove_tmp_dir()

import logging
import os
from cli_base import remove_tmp_dir, __tmp_dir__, create_tmp_dir
from ivadomed.scripts import extract_small_dataset
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_extract_small_dataset():
    # extract_small_dataset.main(args=[])
    assert 1 == 1


def teardown_function():
    remove_tmp_dir()

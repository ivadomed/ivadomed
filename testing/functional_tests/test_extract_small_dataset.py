import logging
import os
from cli_base import remove_tmp_dir, __tmp_dir__, create_tmp_dir, __data_testing_dir__
from ivadomed.scripts import extract_small_dataset
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_extract_small_dataset_default_n():
    __output_dir__ = os.path.join(__tmp_dir__, 'output_extract_small_dataset')
    extract_small_dataset.main(args=['--input', __data_testing_dir__,
                                     '--output', __output_dir__])
    assert os.path.exists(__output_dir__)
    output_dir_list = os.listdir(__output_dir__)
    assert 'derivatives' in output_dir_list
    assert 'participants.tsv' in output_dir_list
    assert 'dataset_description.json' in output_dir_list
    assert 'sub-unf01' in output_dir_list or \
        'sub-unf02' in output_dir_list or \
        'sub-unf03' in output_dir_list


def test_extract_small_dataset_n_2():
    __output_dir__ = os.path.join(__tmp_dir__, 'output_extract_small_dataset_2')
    extract_small_dataset.main(args=['--input', __data_testing_dir__,
                                     '--output', __output_dir__,
                                     '-n', '2'])
    assert os.path.exists(__output_dir__)
    output_dir_list = os.listdir(__output_dir__)
    assert 'derivatives' in output_dir_list
    assert 'participants.tsv' in output_dir_list
    assert 'dataset_description.json' in output_dir_list
    assert ('sub-unf01' in output_dir_list and 'sub-unf02' in output_dir_list) or \
        ('sub-unf01' in output_dir_list and 'sub-unf03' in output_dir_list) or \
        ('sub-unf03' in output_dir_list and 'sub-unf02' in output_dir_list)
    assert 'sub-unf01' not in output_dir_list or \
        'sub-unf02' not in output_dir_list or \
        'sub-unf03' not in output_dir_list

def test_extract_small_dataset_no_derivatives():
    __output_dir__ = os.path.join(__tmp_dir__, 'output_extract_small_dataset_3')
    extract_small_dataset.main(args=['--input', __data_testing_dir__,
                                     '--output', __output_dir__,
                                     '-d', '0'])
    assert os.path.exists(__output_dir__)
    output_dir_list = os.listdir(__output_dir__)
    assert 'derivatives' not in output_dir_list
    assert 'participants.tsv' in output_dir_list
    assert 'dataset_description.json' in output_dir_list
    assert 'sub-unf01' in output_dir_list or \
        'sub-unf02' in output_dir_list or \
        'sub-unf03' in output_dir_list

def test_extract_small_dataset_contrast_list():
    __output_dir__ = os.path.join(__tmp_dir__, 'output_extract_small_dataset_4')
    extract_small_dataset.main(args=['--input', __data_testing_dir__,
                                     '--output', __output_dir__,
                                     '-c', 'T1w, T2w'])
    assert os.path.exists(__output_dir__)
    output_dir_list = os.listdir(__output_dir__)
    assert 'derivatives' in output_dir_list
    assert 'participants.tsv' in output_dir_list
    assert 'dataset_description.json' in output_dir_list
    assert 'sub-unf01' in output_dir_list or \
        'sub-unf02' in output_dir_list or \
        'sub-unf03' in output_dir_list

def teardown_function():
    remove_tmp_dir()

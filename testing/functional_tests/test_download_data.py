import logging
import pytest
import os
from functional_tests.t_utils import remove_dataset, __tmp_dir__, create_tmp_dir, remove_tmp_dir
from ivadomed.scripts import download_data
from ivadomed.utils import ArgParseException
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir(copy_data_testing_dir=False)


def test_download_data():
    for dataset in download_data.DICT_URL:
        output_folder = os.path.join(__tmp_dir__, dataset)
        download_data.main(args=['-d', dataset,
                                 '-o', output_folder])
        assert os.path.exists(output_folder)
        remove_dataset(dataset=dataset)


def test_download_data_no_dataset_specified():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        download_data.main()


def teardown_function():
    remove_tmp_dir()

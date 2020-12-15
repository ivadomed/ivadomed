import logging
import pytest
import os
from cli_base import __test_dir__, remove_dataset
from ivadomed.scripts import download_data
from ivadomed.utils import ArgParseException
logger = logging.getLogger(__name__)


def test_download_data():
    for dataset in download_data.DICT_URL:
        output_folder = os.path.join(__test_dir__, dataset)
        download_data.main(args=['-d', dataset,
                                 '-o', output_folder])
        assert os.path.exists(output_folder)
        remove_dataset(dataset=dataset)


def test_download_data_no_dataset_specified():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        download_data.main()


def teardown_function():
    for dataset in download_data.DICT_URL:
        remove_dataset(dataset=dataset)

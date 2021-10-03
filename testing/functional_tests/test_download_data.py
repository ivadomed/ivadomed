import logging
import pytest
import os
from testing.functional_tests.t_utils import __tmp_dir__, create_tmp_dir
from testing.common_testing_util import remove_dataset, remove_tmp_dir
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
    with pytest.raises(Exception) as e_info:
        download_data.main()
    if e_info.type == SystemExit:
        assert e_info.value == 0
    elif e_info.type == ArgParseException:
        assert e_info.match(r"Error parsing args")



def teardown_function():
    remove_tmp_dir()

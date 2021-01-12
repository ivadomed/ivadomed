import logging
import pytest
import os
from cli_base import remove_tmp_dir, __tmp_dir__, create_tmp_dir, __data_testing_dir__
from ivadomed.scripts import convert_to_onnx
from ivadomed.utils import ArgParseException
logger = logging.getLogger(__name__)

__model_path__ = os.path.join(__data_testing_dir__, 'spinegeneric_model.pt')


def setup_function():
    create_tmp_dir()


def test_convert_to_onnx():
    convert_to_onnx.main(args=['-m', f'{__model_path__}', '-d', '2'])
    assert os.path.exists(os.path.join(__data_testing_dir__, 'spinegeneric_model.onnx'))


def test_convert_to_onnx_no_model():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        convert_to_onnx.main(args=['-d', '2'])


def test_convert_to_onnx_no_dimension():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        convert_to_onnx.main(args=['-m', f'{__model_path__}'])


def teardown_function():
    remove_tmp_dir()

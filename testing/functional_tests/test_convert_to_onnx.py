import logging
import pytest
import os
from cli_base import download_dataset, __data_testing_dir__, remove_dataset
from ivadomed.scripts import convert_to_onnx
from ivadomed.utils import ArgParseException
logger = logging.getLogger(__name__)


def setup_function():
    download_dataset()


def test_convert_to_onnx():
    model_path = os.path.join(__data_testing_dir__, 'spinegeneric_model.pt')
    convert_to_onnx.main(args=['-m', f'{model_path}', '-d', '2'])
    assert os.path.exists(os.path.join(__data_testing_dir__, 'spinegeneric_model.onnx'))


def test_convert_to_onnx_no_model():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        convert_to_onnx.main(args=['-d', '2'])


def test_convert_to_onnx_no_dimension():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        model_path = os.path.join(__data_testing_dir__, 'spinegeneric_model.pt')
        convert_to_onnx.main(args=['-m', f'{model_path}'])


def teardown_function():
    remove_dataset()

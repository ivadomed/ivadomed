import logging
import pytest
import os
from cli_base import remove_tmp_dir, __tmp_dir__, create_tmp_dir
from ivadomed.scripts import convert_to_onnx
from ivadomed.utils import ArgParseException
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_convert_to_onnx():
    model_path = os.path.join(__tmp_dir__, 'spinegeneric_model.pt')
    convert_to_onnx.main(args=['-m', f'{model_path}', '-d', '2'])
    assert os.path.exists(os.path.join(__tmp_dir__, 'spinegeneric_model.onnx'))


def test_convert_to_onnx_no_model():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        convert_to_onnx.main(args=['-d', '2'])


def test_convert_to_onnx_no_dimension():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        model_path = os.path.join(__tmp_dir__, 'spinegeneric_model.pt')
        convert_to_onnx.main(args=['-m', f'{model_path}'])


def teardown_function():
    remove_tmp_dir()

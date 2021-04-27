import logging
import os
from testing.functional_tests.t_utils import __tmp_dir__, create_tmp_dir, download_functional_test_files, \
    generate_labels
from testing.common_testing_util import remove_tmp_dir
from ivadomed.scripts import visualize_transforms

logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()
    generate_labels()


def test_visualize_transforms_n_1(download_functional_test_files):
    __data_testing_dir__ = os.path.join(__tmp_dir__, "data_functional_testing")
    __input_file__ = os.path.join(__data_testing_dir__, 'sub-unf01/anat/sub-unf01_T1w.nii.gz')
    __output_dir__ = os.path.join(__tmp_dir__, "output_visualize_transforms_n_1")
    __config_file__ = os.path.join(__data_testing_dir__, "model_config.json")
    __label_file__ = os.path.join(__data_testing_dir__,
                                  'derivatives/labels/sub-test001/anat/sub-unf01_T1w_seg-manual.nii.gz')
    visualize_transforms.main(args=['--input', __input_file__,
                                    '--output', __output_dir__,
                                    '--config', __config_file__,
                                    '-r', __label_file__])
    assert os.path.exists(__output_dir__)
    output_files = os.listdir(__output_dir__)
    assert len(output_files) == 5
    for output_file in output_files:
        assert "Resample" in output_file
        assert "slice" in output_file
        assert ".png" in output_file


def test_visualize_transforms_n_2(download_functional_test_files):
    __data_testing_dir__ = os.path.join(__tmp_dir__, "data_functional_testing")
    __input_file__ = os.path.join(__data_testing_dir__, 'sub-unf01/anat/sub-unf01_T1w.nii.gz')
    __output_dir__ = os.path.join(__tmp_dir__, "output_visualize_transforms_n_2")
    __config_file__ = os.path.join(__data_testing_dir__, "model_config.json")
    __label_file__ = os.path.join(__data_testing_dir__,
                                  'derivatives/labels/sub-test001/anat/sub-unf01_T1w_seg-manual.nii.gz')
    visualize_transforms.main(args=['--input', __input_file__,
                                    '--output', __output_dir__,
                                    '--config', __config_file__,
                                    '-r', __label_file__,
                                    '-n', '2'])
    assert os.path.exists(__output_dir__)
    output_files = os.listdir(__output_dir__)
    assert len(output_files) == 10
    for output_file in output_files:
        assert "Resample" in output_file
        assert "slice" in output_file
        assert ".png" in output_file


def teardown_function():
    remove_tmp_dir()

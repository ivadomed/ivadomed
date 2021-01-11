import logging
import os
from cli_base import remove_tmp_dir, __tmp_dir__, create_tmp_dir, download_dataset
from ivadomed.scripts import prepare_dataset_vertebral_labeling
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()
    download_dataset("data_functional_testing")


def test_prepare_dataset_vertebral_labeling():
    __input_dir__ = os.path.join(__tmp_dir__, 'data_functional_testing')
    prepare_dataset_vertebral_labeling.main(args=['--path', __input_dir__,
                                                  '--suffix', '_T2w',
                                                  '--aim', '3'])
    assert os.path.exists(os.path.join(
        __input_dir__, "derivatives/labels/sub-unf01/anat/sub-unf01_T2w_mid_heatmap3.nii.gz"))
    assert os.path.exists(os.path.join(
        __input_dir__, "derivatives/labels/sub-unf02/anat/sub-unf02_T2w_mid_heatmap3.nii.gz"))
    assert os.path.exists(os.path.join(
        __input_dir__, "derivatives/labels/sub-unf03/anat/sub-unf03_T2w_mid_heatmap3.nii.gz"))


def teardown_function():
    remove_tmp_dir()

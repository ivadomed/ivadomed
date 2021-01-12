import logging
import os
from t_utils import remove_tmp_dir, __tmp_dir__, create_tmp_dir, __data_testing_dir__
from ivadomed.scripts import prepare_dataset_vertebral_labeling
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_prepare_dataset_vertebral_labeling():
    prepare_dataset_vertebral_labeling.main(args=['--path', __data_testing_dir__,
                                                  '--suffix', '_T2w',
                                                  '--aim', '3'])
    assert os.path.exists(os.path.join(
        __data_testing_dir__, "derivatives/labels/sub-unf01/anat/sub-unf01_T2w_mid_heatmap3.nii.gz"))
    assert os.path.exists(os.path.join(
        __data_testing_dir__, "derivatives/labels/sub-unf02/anat/sub-unf02_T2w_mid_heatmap3.nii.gz"))
    assert os.path.exists(os.path.join(
        __data_testing_dir__, "derivatives/labels/sub-unf03/anat/sub-unf03_T2w_mid_heatmap3.nii.gz"))


def teardown_function():
    remove_tmp_dir()

import logging
import os
from testing.functional_tests.t_utils import __tmp_dir__, create_tmp_dir, download_functional_test_files
from testing.common_testing_util import remove_tmp_dir
from ivadomed.scripts import training_curve
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_training_curve(download_functional_test_files):
    __data_testing_dir__ = os.path.join(__tmp_dir__, "data_functional_testing")
    __input_dir__ = os.path.join(__data_testing_dir__, 'tensorboard_events')
    __output_dir__ = os.path.join(__tmp_dir__, 'output_training_curve')
    training_curve.main(args=['--input', __input_dir__,
                              '--output', __output_dir__])
    assert os.path.exists(__output_dir__)
    assert os.path.exists(os.path.join(__output_dir__, "accuracy_score.png"))
    assert os.path.exists(os.path.join(__output_dir__, "dice_score.png"))
    assert os.path.exists(os.path.join(__output_dir__, "hausdorff_score.png"))
    assert os.path.exists(os.path.join(__output_dir__, "intersection_over_union.png"))
    assert os.path.exists(os.path.join(__output_dir__, "losses.png"))
    assert os.path.exists(os.path.join(__output_dir__, "multi_class_dice_score.png"))
    assert os.path.exists(os.path.join(__output_dir__, "precision_score.png"))
    assert os.path.exists(os.path.join(__output_dir__, "recall_score.png"))
    assert os.path.exists(os.path.join(__output_dir__, "specificity_score.png"))
    assert os.path.exists(os.path.join(__output_dir__, "tensorboard_events_training_values.csv"))


def teardown_function():
    remove_tmp_dir()

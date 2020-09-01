import ivadomed.metrics as imed_metrics
import numpy as np
import pytest
import os


@pytest.mark.parametrize("image", np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]]))
def test_multi_class_dice_score(image):
    results = imed_metrics.multi_class_dice_score(image, image)
    assert results == 1


@pytest.mark.parametrize("image", np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]]))
def test_mse(image):
    results = imed_metrics.mse(image, image)
    assert results == 0.0


@pytest.mark.parametrize("image", np.array([[[[1, 1], [1, 1]], [[0, 0], [0, 0]]]]))
def test_haussdorf_4d(image):
    results = imed_metrics.hausdorff_score(image, image)
    assert results == 0.0


# We test error case for the metrics
@pytest.mark.parametrize("image", np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]))
@pytest.mark.parametrize("image_2", np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]))
def test_err_prec(image, image_2):
    results = imed_metrics.precision_score(image, image_2)
    assert results == 0.0


@pytest.mark.parametrize("image", np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]))
@pytest.mark.parametrize("image_2", np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]))
def test_err_rec(image, image_2):
    results = imed_metrics.recall_score(image, image_2, err_value=1)
    assert results == 1


@pytest.mark.parametrize("image", np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]))
@pytest.mark.parametrize("image_2", np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]))
def test_err_spec(image, image_2):
    results = imed_metrics.specificity_score(image, image_2, err_value=12)
    assert results == 12


@pytest.mark.parametrize("image", np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]))
@pytest.mark.parametrize("image_2", np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]))
def test_err_iou(image, image_2):
    results = imed_metrics.intersection_over_union(image, image_2, err_value=12)
    assert results == 12


# we test wether the ploting code run or not
def test_plot_roc_curve():
    tpr = [0, 0.1, 0.5, 0.6, 0.9]
    fpr = [1, 0.8, 0.5, 0.3, 0.1]
    opt_thr_idx = 3
    imed_metrics.plot_roc_curve(tpr, fpr, opt_thr_idx, "roc_test.png")
    assert os.path.isfile("roc_test.png")


def test_dice_plot():
    thr_list = [0.1, 0.3, 0.5, 0.7]
    dice_list = [0.6, 0.7, 0.8, 0.75]
    imed_metrics.plot_dice_thr(thr_list, dice_list, 2, "test_dice.png")
    assert os.path.isfile("test_dice.png")

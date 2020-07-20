import ivadomed.metrics as imed_metrics
import numpy as np


def test_multi_class_dice_score():
    # create fake image
    image = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
    results = imed_metrics.multi_class_dice_score(image, image)


def test_mse():
    # create fake image
    image = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
    results = imed_metrics.mse(image, image)


def test_haussdorf_4d():
    # create fake image
    image = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
    results = imed_metrics.hausdorff_score(image, image)


def test_err_prec():
    # create fake image
    image = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    image_2 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    results = imed_metrics.precision_score(image, image_2)
    assert results == 0.0


def test_err_rec():
    # create fake image
    image = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    image_2 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    results = imed_metrics.recall_score(image, image_2, err_value=1)
    assert results == 1


def test_err_spec():
    # create fake image
    image = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    image_2 = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    results = imed_metrics.specificity_score(image, image_2, err_value=12)
    assert results == 12


def test_err_iou():
    image = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    image_2 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    results = imed_metrics.intersection_over_union(image, image_2, err_value=12)
    assert results == 12





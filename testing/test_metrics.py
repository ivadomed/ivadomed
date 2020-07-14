import ivadomed.metrics as imed_metrics
import numpy as np


def test_multi_class_dice_score():
    # create fake image
    image = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
    results = imed_metrics.multi_class_dice_score(image, image)





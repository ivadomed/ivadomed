from ivadomed import visualize as imed_visualize
import numpy as np
import torch


def test_save_rbg():
    # Create image with shape HxWxDxC
    image = [[[[0 for i in range(10)] for i in range(10)] for i in range(4)] for i in range(3)]
    for i in range(3):
        image[0][2][5][i] = 1
        image[1][2][5][i+3] = 1
        image[2][2][5][i + 6] = 1
    image_n = np.array(image)
    imed_visualize.save_color_labels(image_n, False,
                                 "testing_data/derivatives/labels/sub-unf01/anat/sub-unf01_T2w_lesion-manual.nii.gz",
                                 "rgb_test.nii.gz", 0)


def test_rbg_conversion():
    # Create image with shape HxWxn_classxbatch_size
    image = [[[[0 for i in range(10)] for i in range(10)] for i in range(3)] for i in range(3)]
    for i in range(3):
        image[0][2][5][i] = 1
        image[1][2][5][i + 3] = 1
        image[2][2][5][i + 6] = 1
    image_n = np.array(image)
    tensor_multi = torch.tensor(image_n)
    imed_visualize.convert_labels_to_RGB(tensor_multi)

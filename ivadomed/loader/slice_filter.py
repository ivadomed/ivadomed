import torch
import numpy as np
from ivadomed import utils as imed_utils


class SliceFilter(object):
    """Filter 2D slices or patches from dataset.

    If a slice or 2D patch does not meet certain conditions, it is discarded from the dataset.

    Args:
        filter_empty_mask (bool): If True, slices where all voxel labels are zeros are discarded.
        filter_absent_class (bool): If True, slices where all voxel labels are zero for one or more classes are discarded.
        filter_empty_input (bool): If True, slices where all voxel intensities are zeros are discarded.
        filter_classification (bool): If True, slices where all images fail a custom classifier filter are discarded.
        filter_empty_mask_patch (bool): If True, 2D patches where all voxel labels are zeros are discarded at training time.
        filter_absent_class_patch (bool): If True, 2D patches where all voxel labels are zero for one or more classes are discarded at training time.
        filter_empty_input_patch (bool): If True, 2D patches where all voxel intensities are zeros are discarded at training time.
        device (torch.device): Indicates the CPU or GPU ID.
        cuda_available (bool): If True, CUDA is available.
        is_train (bool): Indicates if at training time.

    Attributes:
        filter_empty_mask (bool): If True, slices where all voxel labels are zeros are discarded.
        filter_absent_class (bool): If True, slices where all voxel labels are zero for one or more classes are discarded.
        filter_empty_input (bool): If True, slices where all voxel intensities are zeros are discarded.
        filter_classification (bool): If True, slices where all images fail a custom classifier filter are discarded.
        filter_empty_mask_patch (bool): If True, 2D patches where all voxel labels are zeros are discarded at training time.
        filter_absent_class_patch (bool): If True, 2D patches where all voxel labels are zero for one or more classes are discarded at training time.
        filter_empty_input_patch (bool): If True, 2D patches where all voxel intensities are zeros are discarded at training time.
        device (torch.device): Indicates the CPU or GPU ID.
        cuda_available (bool): If True, CUDA is available.
        is_train (bool): Indicates if at training time.

    """

    def __init__(self, filter_empty_mask=True,
                 filter_absent_class=False,
                 filter_empty_input=True,
                 filter_classification=False,
                 filter_empty_mask_patch = False,
                 filter_absent_class_patch=False,
                 filter_empty_input_patch=False,
                 classifier_path=None, device=None, cuda_available=None, is_train=False):
        self.filter_empty_mask = filter_empty_mask
        self.filter_absent_class = filter_absent_class
        self.filter_empty_input = filter_empty_input
        self.filter_classification = filter_classification
        self.filter_empty_mask_patch = filter_empty_mask_patch
        self.filter_absent_class_patch = filter_absent_class_patch
        self.filter_empty_input_patch = filter_empty_input_patch
        self.device = device
        self.cuda_available = cuda_available
        self.is_train = is_train

        if self.filter_classification:
            if cuda_available:
                self.classifier = torch.load(classifier_path, map_location=device)
            else:
                self.classifier = torch.load(classifier_path, map_location='cpu')

    def __call__(self, sample, is_2d_patch=False):
        input_data, gt_data = sample['input'], sample['gt']

        if is_2d_patch:
            if self.is_train:
                if self.filter_empty_mask_patch:
                    # Filter 2D patches at training time that do not have ANY ground truth (i.e. all masks are empty)
                    if not np.any(gt_data):
                        return False

                if self.filter_absent_class_patch:
                    # Filter 2D patches at training time that have absent classes (i.e. one or more masks are empty)
                    if not np.all([np.any(mask) for mask in gt_data]):
                        return False

                if self.filter_empty_input_patch:
                    # Filter set of 2D patches at training time if one of them is empty or filled with constant value (i.e. std == 0)
                    if np.any([img.std() == 0 for img in input_data]):
                        return False
        else:
            if self.filter_empty_mask:
                # Filter slices that do not have ANY ground truth (i.e. all masks are empty)
                if not np.any(gt_data):
                    return False

            if self.filter_absent_class:
                # Filter slices that have absent classes (i.e. one or more masks are empty)
                if not np.all([np.any(mask) for mask in gt_data]):
                    return False

            if self.filter_empty_input:
                # Filter set of images if one of them is empty or filled with constant value (i.e. std == 0)
                if np.any([img.std() == 0 for img in input_data]):
                    return False

            if self.filter_classification:
                if not np.all([int(
                        self.classifier(
                            imed_utils.cuda(torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(0),
                                            self.cuda_available))) for img in input_data]):
                    return False

        return True

import copy
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from ivadomed import transforms as imed_transforms, postprocessing as imed_postpro
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.loader.utils import dropout_input
from ivadomed.loader.segmentation_pair import SegmentationPair
from ivadomed.object_detection import utils as imed_obj_detect
from ivadomed.keywords import MetadataKW


class MRI3DSubVolumeSegmentationDataset(Dataset):
    """This is a class for 3D segmentation dataset. This class splits the initials volumes in several
    subvolumes. Each subvolumes will be of the sizes of the length parameter.

    This class also implement a stride parameter corresponding to the amount of voxels subvolumes are translated in
    each dimension at every iteration.

    Be careful, the input's dimensions should be compatible with the given
    lengths and strides. This class doesn't handle missing dimensions.

    Args:
        filename_pairs (list): A list of tuples in the format (input filename, ground truth filename).
        transform (Compose): Transformations to apply.
        length (tuple): Size of each dimensions of the subvolumes, length equals 3.
        stride (tuple): Size of the overlapping per subvolume and dimensions, length equals 3.
        slice_axis (int): Indicates the axis used to extract slices: "axial": 2, "sagittal": 0, "coronal": 1.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
        truths are thresholded (0.5) after the data augmentation operations.
        is_input_dropout (bool): Return input with missing modalities.
    """

    def __init__(self, filename_pairs, transform=None, length=(64, 64, 64), stride=(0, 0, 0), slice_axis=0,
                 task="segmentation", soft_gt=False, is_input_dropout=False):
        self.filename_pairs = filename_pairs
        self.handlers = []
        self.indexes = []
        self.length = length
        self.stride = stride
        self.prepro_transforms, self.transform = transform
        self.slice_axis = slice_axis
        self.has_bounding_box = True
        self.task = task
        self.soft_gt = soft_gt
        self.is_input_dropout = is_input_dropout

        self._load_filenames()
        self._prepare_indices()

    def _load_filenames(self):
        """Load preprocessed pair data (input and gt) in handler."""
        for input_filename, gt_filename, roi_filename, metadata in self.filename_pairs:
            segpair = SegmentationPair(input_filename, gt_filename, metadata=metadata, slice_axis=self.slice_axis,
                                       soft_gt=self.soft_gt)
            input_data, gt_data = segpair.get_pair_data()
            metadata = segpair.get_pair_metadata()
            seg_pair = {
                'input': input_data,
                'gt': gt_data,
                MetadataKW.INPUT_METADATA: metadata[MetadataKW.INPUT_METADATA],
                MetadataKW.GT_METADATA: metadata[MetadataKW.GT_METADATA]
            }

            self.has_bounding_box = imed_obj_detect.verify_metadata(seg_pair, self.has_bounding_box)
            if self.has_bounding_box:
                self.prepro_transforms = imed_obj_detect.adjust_transforms(self.prepro_transforms, seg_pair,
                                                                           length=self.length,
                                                                           stride=self.stride)
            seg_pair, roi_pair = imed_transforms.apply_preprocessing_transforms(self.prepro_transforms,
                                                                                seg_pair=seg_pair)

            for metadata in seg_pair[MetadataKW.INPUT_METADATA]:
                metadata[MetadataKW.INDEX_SHAPE] = seg_pair['input'][0].shape
            self.handlers.append((seg_pair, roi_pair))

    def _prepare_indices(self):
        """Stores coordinates of subvolumes for training."""
        for i in range(0, len(self.handlers)):
            segpair, _ = self.handlers[i]
            input_img = self.handlers[i][0]['input']
            shape = input_img[0].shape

            if ((shape[0] - self.length[0]) % self.stride[0]) != 0 or self.length[0] % 16 != 0 or shape[0] < \
                    self.length[0] \
                    or ((shape[1] - self.length[1]) % self.stride[1]) != 0 or self.length[1] % 16 != 0 or shape[1] < \
                    self.length[1] \
                    or ((shape[2] - self.length[2]) % self.stride[2]) != 0 or self.length[2] % 16 != 0 or shape[2] < \
                    self.length[2]:
                raise RuntimeError('Input shape of each dimension should be a \
                                    multiple of length plus 2 * padding and a multiple of 16.')

            for x in range(0, (shape[0] - self.length[0]) + 1, self.stride[0]):
                for y in range(0, (shape[1] - self.length[1]) + 1, self.stride[1]):
                    for z in range(0, (shape[2] - self.length[2]) + 1, self.stride[2]):
                        self.indexes.append({
                            'x_min': x,
                            'x_max': x + self.length[0],
                            'y_min': y,
                            'y_max': y + self.length[1],
                            'z_min': z,
                            'z_max': z + self.length[2],
                            'handler_index': i})

    def __len__(self):
        """Return the dataset size. The number of subvolumes."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair subvolume (input, ground truth).

        Args:
            index (int): Subvolume index.
        """

        # copy.deepcopy is used to have different coordinates for reconstruction for a given handler,
        # to allow a different rater at each iteration of training, and to clean transforms params from previous
        # transforms i.e. remove params from previous iterations so that the coming transforms are different
        coord = self.indexes[index]
        seg_pair, _ = copy.deepcopy(self.handlers[coord['handler_index']])

        # In case multiple raters
        if seg_pair['gt'] and isinstance(seg_pair['gt'][0], list):
            # Randomly pick a rater
            idx_rater = random.randint(0, len(seg_pair['gt'][0]) - 1)
            # Use it as ground truth for this iteration
            # Note: in case of multi-class: the same rater is used across classes
            for idx_class in range(len(seg_pair['gt'])):
                seg_pair['gt'][idx_class] = seg_pair['gt'][idx_class][idx_rater]
                seg_pair['gt_metadata'][idx_class] = seg_pair['gt_metadata'][idx_class][idx_rater]

        metadata_input = seg_pair['input_metadata'] if seg_pair['input_metadata'] is not None else []
        metadata_gt = seg_pair['gt_metadata'] if seg_pair['gt_metadata'] is not None else []

        # Run transforms on images
        stack_input, metadata_input = self.transform(sample=seg_pair['input'],
                                                     metadata=metadata_input,
                                                     data_type="im")
        # Update metadata_gt with metadata_input
        metadata_gt = imed_loader_utils.update_metadata(metadata_input, metadata_gt)

        # Run transforms on images
        stack_gt, metadata_gt = self.transform(sample=seg_pair['gt'],
                                               metadata=metadata_gt,
                                               data_type="gt")
        # Make sure stack_gt is binarized
        if stack_gt is not None and not self.soft_gt:
            stack_gt = imed_postpro.threshold_predictions(stack_gt, thr=0.5).astype(np.uint8)

        shape_x = coord["x_max"] - coord["x_min"]
        shape_y = coord["y_max"] - coord["y_min"]
        shape_z = coord["z_max"] - coord["z_min"]

        # add coordinates to metadata to reconstruct volume
        for metadata in metadata_input:
            metadata[MetadataKW.COORD] = [coord["x_min"], coord["x_max"], coord["y_min"], coord["y_max"], coord["z_min"],
                                 coord["z_max"]]

        subvolumes = {
            'input': torch.zeros(stack_input.shape[0], shape_x, shape_y, shape_z),
            'gt': torch.zeros(stack_gt.shape[0], shape_x, shape_y, shape_z) if stack_gt is not None else None,
            MetadataKW.INPUT_METADATA: metadata_input,
            MetadataKW.GT_METADATA: metadata_gt
        }

        for _ in range(len(stack_input)):
            subvolumes['input'] = stack_input[:,
                                  coord['x_min']:coord['x_max'],
                                  coord['y_min']:coord['y_max'],
                                  coord['z_min']:coord['z_max']]

        # Input-level dropout to train with missing modalities
        if self.is_input_dropout:
            subvolumes = dropout_input(subvolumes)

        if stack_gt is not None:
            for _ in range(len(stack_gt)):
                subvolumes['gt'] = stack_gt[:,
                                   coord['x_min']:coord['x_max'],
                                   coord['y_min']:coord['y_max'],
                                   coord['z_min']:coord['z_max']]

        return subvolumes

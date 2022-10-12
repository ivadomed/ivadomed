import copy
import random
from pathlib import Path
import pickle
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger

from ivadomed import transforms as imed_transforms, postprocessing as imed_postpro
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.loader.utils import dropout_input, create_temp_directory, get_obj_size
from ivadomed.loader.segmentation_pair import SegmentationPair
from ivadomed.object_detection import utils as imed_obj_detect
from ivadomed.keywords import MetadataKW, SegmentationDatasetKW, SegmentationPairKW
from ivadomed.utils import get_timestamp, get_system_memory


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
        disk_cache (bool): set whether all input data should be cached in local folders to allow faster subsequent
        reloading and bypass memory cap.
    """

    def __init__(self, filename_pairs, transform=None, length=(64, 64, 64), stride=(0, 0, 0), slice_axis=0,
                 task="segmentation", soft_gt=False, is_input_dropout=False, disk_cache=True):
        self.filename_pairs = filename_pairs

        # could be a list of tuple of objects OR path objects to the actual disk equivalent.
        # behaves differently depend on if self.cache is set to true or not.
        self.handlers: List[tuple] = []

        self.indexes: list = []
        self.length = length
        self.stride = stride
        self.prepro_transforms, self.transform = transform
        self.slice_axis = slice_axis
        self.has_bounding_box: bool = True
        self.task = task
        self.soft_gt = soft_gt
        self.is_input_dropout = is_input_dropout
        self.disk_cache: bool = disk_cache

        self._load_filenames()
        self._prepare_indices()

    def _load_filenames(self):
        """Load preprocessed pair data (input and gt) in handler."""
        path_temp: Path = Path(create_temp_directory())

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

            # First time detemine cache automatically IF not specified. Otherwise, use the cache specified.
            if self.disk_cache is None:
                self.disk_cache = self.determine_cache_need(seg_pair, roi_pair)

            if self.disk_cache:
                # Write SegPair and ROIPair to disk cache with timestamp to avoid collisions
                path_cache_seg_pair = path_temp / f'seg_pair_{get_timestamp()}.pkl'
                with path_cache_seg_pair.open(mode='wb') as f:
                    pickle.dump(seg_pair, f)

                path_cache_roi_pair = path_temp / f'roi_pair_{get_timestamp()}.pkl'
                with path_cache_roi_pair.open(mode='wb') as f:
                    pickle.dump(roi_pair, f)
                self.handlers.append((path_cache_seg_pair, path_cache_roi_pair))

            else:
                # self.handler is now a list of a FILES instead of actual data to prevent this list from taking up too much
                # memory
                self.handlers.append((seg_pair, roi_pair))

    def _prepare_indices(self):
        """Stores coordinates of subvolumes for training."""
        for i in range(0, len(self.handlers)):

            if self.disk_cache:
                with self.handlers[i][0].open(mode='rb') as f:
                    segpair = pickle.load(f)
            else:
                segpair = self.handlers[i][0]

            input_img = segpair.get('input')
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

    def __getitem__(self, subvolume_index: int):
        """Return the specific index pair subvolume (input, ground truth).

        Args:
            subvolume_index (int): Subvolume index.
        """

        # copy.deepcopy is used to have different coordinates for reconstruction for a given handler,
        # to allow a different rater at each iteration of training, and to clean transforms params from previous
        # transforms i.e. remove params from previous iterations so that the coming transforms are different

        # Get the tuple that defines the boundaries for the subsample
        coord: dict = self.indexes[subvolume_index]
        x_min = coord.get(SegmentationDatasetKW.X_MIN)
        x_max = coord.get(SegmentationDatasetKW.X_MAX)
        y_min = coord.get(SegmentationDatasetKW.Y_MIN)
        y_max = coord.get(SegmentationDatasetKW.Y_MAX)
        z_min = coord.get(SegmentationDatasetKW.Z_MIN)
        z_max = coord.get(SegmentationDatasetKW.Z_MAX)

        # Obtain tuple reference to the pairs of file references
        tuple_seg_roi_pair: tuple = self.handlers[coord.get(SegmentationDatasetKW.HANDLER_INDEX)]

        # Disk Cache handling, either, load the seg_pair, not using ROI pair here.
        if self.disk_cache:
            with tuple_seg_roi_pair[0].open(mode='rb') as f:
                seg_pair = pickle.load(f)
        else:
            seg_pair, _ = copy.deepcopy(tuple_seg_roi_pair)

        # In case multiple raters
        if seg_pair[SegmentationPairKW.GT] and isinstance(seg_pair[SegmentationPairKW.GT][0], list):
            # Randomly pick a rater
            idx_rater = random.randint(0, len(seg_pair[SegmentationPairKW.GT][0]) - 1)
            # Use it as ground truth for this iteration
            # Note: in case of multi-class: the same rater is used across classes
            for idx_class in range(len(seg_pair[SegmentationPairKW.GT])):
                seg_pair[SegmentationPairKW.GT][idx_class] = seg_pair[SegmentationPairKW.GT][idx_class][idx_rater]
                seg_pair[SegmentationPairKW.GT_METADATA][idx_class] = seg_pair[SegmentationPairKW.GT_METADATA][idx_class][idx_rater]

        if seg_pair[SegmentationPairKW.INPUT_METADATA]:
            metadata_input = seg_pair[SegmentationPairKW.INPUT_METADATA]
        else:
            metadata_input = []

        if seg_pair[SegmentationPairKW.GT_METADATA]:
            metadata_gt = seg_pair[SegmentationPairKW.GT_METADATA]
        else:
            metadata_gt = []

        # Extract image and gt slices or patches from coordinates
        stack_input = np.asarray(seg_pair[SegmentationPairKW.INPUT])[
                      :,
                      x_min:x_max,
                      y_min:y_max,
                      z_min:z_max
            ]

        if seg_pair[SegmentationPairKW.GT]:
            stack_gt = np.asarray(seg_pair[SegmentationPairKW.GT])[
                       :,
                       x_min:x_max,
                       y_min:y_max,
                       z_min:z_max
            ]
        else:
            stack_gt = []

        # Run transforms on image slices
        stack_input, metadata_input = self.transform(sample=stack_input,
                                                     metadata=metadata_input,
                                                     data_type="im")
        # Update metadata_gt with metadata_input
        metadata_gt = imed_loader_utils.update_metadata(metadata_input, metadata_gt)

        # Run transforms on gt slices
        stack_gt, metadata_gt = self.transform(sample=stack_gt,
                                               metadata=metadata_gt,
                                               data_type="gt")
        # Make sure stack_gt is binarized
        if stack_gt is not None and not self.soft_gt:
            stack_gt = imed_postpro.threshold_predictions(stack_gt, thr=0.5).astype(np.uint8)

        shape_x = x_max - x_min
        shape_y = y_max - y_min
        shape_z = z_max - z_min

        # add coordinates to metadata to reconstruct volume
        for metadata in metadata_input:
            metadata[MetadataKW.COORD] = [
                x_min, x_max,
                y_min, y_max,
                z_min, z_max,
            ]

        subvolumes = {
            SegmentationPairKW.INPUT: stack_input,
            SegmentationPairKW.GT: stack_gt,
            MetadataKW.INPUT_METADATA: metadata_input,
            MetadataKW.GT_METADATA: metadata_gt
        }

        # Input-level dropout to train with missing modalities
        if self.is_input_dropout:
            subvolumes = dropout_input(subvolumes)

        return subvolumes

    def determine_cache_need(self, seg_pair: dict, roi_pair: dict):
        """
        When Cache flag is not explicitly set, determine whether to cache the data or not
        Args:
            seg_pair: an EXAMPLE, typical seg_pair object
            roi_pair: an EXAMPLE, typical seg_pair object

        Returns:

        """
        size_seg_pair_in_bytes = get_obj_size(seg_pair)
        size_roi_pair_in_bytes = get_obj_size(roi_pair)

        optimal_ram_limit = get_system_memory() * 0.5

        # Size limit: 4GB GPU RAM, keep in mind tranform etc might take MORE!
        size_estimated_dataset_GB = (size_seg_pair_in_bytes + size_roi_pair_in_bytes) * len(self.filename_pairs) / 1024 ** 3
        if size_estimated_dataset_GB > optimal_ram_limit:
            logger.info(f"Estimated 3D dataset size is {size_estimated_dataset_GB} GB, which is larger than {optimal_ram_limit} GB. Auto "
                        f"enabling cache.")
            self.disk_cache = True
            return True
        else:
            logger.info(f"Estimated 3D dataset size is {size_estimated_dataset_GB} GB, which is smaller than {optimal_ram_limit} GB. File "
                        f"cache will not be used")
            self.disk_cache = False
            return False

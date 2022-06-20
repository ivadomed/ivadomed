import copy
import random
from pathlib import Path
import pickle

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger

from ivadomed import transforms as imed_transforms, postprocessing as imed_postpro
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.loader.utils import dropout_input, get_obj_size, create_temp_directory
from ivadomed.loader.segmentation_pair import SegmentationPair
from ivadomed.object_detection import utils as imed_obj_detect
from ivadomed.keywords import ROIParamsKW, MetadataKW
from ivadomed.utils import get_timestamp, get_system_memory


class MRI2DSegmentationDataset(Dataset):
    """Generic class for 2D (slice-wise) segmentation dataset.

    Args:
        filename_pairs (list): a list of tuples in the format (input filename list containing all modalities,ground \
            truth filename, ROI filename, metadata).
        length (list): Size of each dimensions of the patches, length equals 0 (no patching) or 2 (2d patching).
        stride (list): Size of the pixels' shift between patches, length equals 0 (no patching) or 2 (2d patching).
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        nibabel_cache (bool): if the data should be cached in memory or not.
        transform (torchvision.Compose): transformations to apply.
        slice_filter_fn (SliceFilter): SliceFilter object containing Slice filter parameters.
        patch_filter_fn (PatchFilter): PatchFilter object containing Patch filter parameters.
        task (str): choice between segmentation or classification. If classification: GT is discrete values, \
            If segmentation: GT is binary mask.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
        truths are thresholded (0.5) after the data augmentation operations.
        is_input_dropout (bool): Return input with missing modalities.

    Attributes:
        indexes (list): List of indices corresponding to each slice or patch in the dataset.
        handlers (list): List of indices corresponding to each slice in the dataset, used for indexing patches.
        filename_pairs (list): List of tuples in the format (input filename list containing all modalities,ground \
            truth filename, ROI filename, metadata).
        length (list): Size of each dimensions of the patches, length equals 0 (no patching) or 2 (2d patching).
        stride (list): Size of the pixels' shift between patches, length equals 0 (no patching) or 2 (2d patching).
        is_2d_patch (bool): True if length in model params.
        prepro_transforms (Compose): Transformations to apply before training.
        transform (Compose): Transformations to apply during training.
        nibabel_cache (bool): determine if the nibabel data object should be cached in memory or not to avoid repetitive
        disk loading
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        slice_filter_fn (SliceFilter): SliceFilter object containing Slice filter parameters.
        patch_filter_fn (PatchFilter): PatchFilter object containing Patch filter parameters.
        n_contrasts (int): Number of input contrasts.
        has_bounding_box (bool): True if bounding box in all metadata, else False.
        task (str): Choice between segmentation or classification. If classification: GT is discrete values, \
            If segmentation: GT is binary mask.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
        truths are thresholded (0.5) after the data augmentation operations.
        slice_filter_roi (bool): Indicates whether a slice filtering is done based on ROI data.
        roi_thr (int): If the ROI mask contains less than this number of non-zero voxels, the slice will be discarded
            from the dataset.
        is_input_dropout (bool): Return input with missing modalities.
        disk_cache (bool): determines whether the items in the segmentation pairs for the entire dataset are cached on
            disk (True) or in memory (False). Default to None to automatically determine based on guesstimated size of
            the entire datasets naively assuming that first image in first volume is representative.

    """

    def __init__(self, filename_pairs, length=None, stride=None, slice_axis=2, nibabel_cache=True, transform=None,
                 slice_filter_fn=None, patch_filter_fn=None, task="segmentation", roi_params=None, soft_gt=False,
                 is_input_dropout=False, disk_cache=None):
        if length is None:
            length = []
        if stride is None:
            stride = []
        self.indexes: list = []
        self.handlers: list = []
        self.filename_pairs = filename_pairs
        self.length = length
        self.stride = stride
        self.is_2d_patch = True if self.length else False
        self.prepro_transforms, self.transform = transform
        self.cache = nibabel_cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.patch_filter_fn = patch_filter_fn
        self.n_contrasts = len(self.filename_pairs[0][0])
        if roi_params is None:
            roi_params = {ROIParamsKW.SUFFIX: None, ROIParamsKW.SLICE_FILTER_ROI: None}
        self.roi_thr = roi_params[ROIParamsKW.SLICE_FILTER_ROI]
        self.slice_filter_roi = roi_params[ROIParamsKW.SUFFIX] is not None and isinstance(self.roi_thr, int)
        self.soft_gt = soft_gt
        self.has_bounding_box = True
        self.task = task
        self.is_input_dropout = is_input_dropout
        self.disk_cache: bool = disk_cache

    def load_filenames(self):
        """Load preprocessed pair data (input and gt) in handler."""
        for input_filenames, gt_filenames, roi_filename, metadata in self.filename_pairs:
            roi_pair = SegmentationPair(input_filenames, roi_filename, metadata=metadata, slice_axis=self.slice_axis,
                                        cache=self.cache, prepro_transforms=self.prepro_transforms)

            seg_pair = SegmentationPair(input_filenames, gt_filenames, metadata=metadata, slice_axis=self.slice_axis,
                                        cache=self.cache, prepro_transforms=self.prepro_transforms,
                                        soft_gt=self.soft_gt)

            input_data_shape, _ = seg_pair.get_pair_shapes()

            path_temp = Path(create_temp_directory())

            for idx_pair_slice in range(input_data_shape[-1]):
                slice_seg_pair = seg_pair.get_pair_slice(idx_pair_slice, gt_type=self.task)
                self.has_bounding_box = imed_obj_detect.verify_metadata(slice_seg_pair, self.has_bounding_box)

                if self.has_bounding_box:
                    self.prepro_transforms = imed_obj_detect.adjust_transforms(self.prepro_transforms, slice_seg_pair)

                if self.slice_filter_fn and not self.slice_filter_fn(slice_seg_pair):
                    continue

                # Note: we force here gt_type=segmentation since ROI slice is needed to Crop the image
                slice_roi_pair = roi_pair.get_pair_slice(idx_pair_slice, gt_type="segmentation")

                if self.slice_filter_roi and imed_loader_utils.filter_roi(slice_roi_pair['gt'], self.roi_thr):
                    continue

                item: Tuple[dict, dict] = imed_transforms.apply_preprocessing_transforms(self.prepro_transforms,
                                                                      slice_seg_pair,
                                                                      slice_roi_pair)
                # Run once code to keep track if disk cache is used
                if self.disk_cache is None:
                    self.determine_cache_need(item, input_data_shape[-1])

                # If is_2d_patch, create handlers list for indexing patch
                if self.is_2d_patch:
                    for metadata in item[0][MetadataKW.INPUT_METADATA]:
                        metadata[MetadataKW.INDEX_SHAPE] = item[0]['input'][0].shape
                    if self.disk_cache:
                        path_item = path_temp / f"item_{get_timestamp()}.pkl"
                        with path_item.open(mode="wb") as f:
                            pickle.dump(item, f)
                        self.handlers.append((path_item))
                    else:
                        self.handlers.append((item))
                # else, append the whole slice to self.indexes
                else:

                    if self.disk_cache:
                        path_item = path_temp / f"item_{get_timestamp()}.pkl"
                        with path_item.open(mode="wb") as f:
                            pickle.dump(item, f)
                        self.indexes.append(path_item)
                    else:
                        self.indexes.append(item)

        # If is_2d_patch, prepare indices of patches
        if self.is_2d_patch:
            self.prepare_indices()

    def prepare_indices(self):
        """Stores coordinates of 2d patches for training."""
        for i in range(0, len(self.handlers)):

            if self.disk_cache:
                with self.handlers[i].open(mode="rb") as f:
                    item = pickle.load(f)
                    primary_handle = item[0]
            else:
                primary_handle = self.handlers[i][0]

            input_img = primary_handle.get('input')
            gt_img = primary_handle.get('gt')
            input_metadata = primary_handle.get('input_metadata')
            gt_metadata = primary_handle.get('gt_metadata')

            shape = input_img[0].shape

            if len(self.length) != 2 or len(self.stride) != 2:
                raise RuntimeError('"length_2D" and "stride_2D" must be of length 2.')
            for length, stride, size in zip(self.length, self.stride, shape):
                if stride > length or stride <= 0:
                    raise RuntimeError('"stride_2D" must be greater than 0 and smaller or equal to "length_2D".')
                if length > size:
                    raise RuntimeError('"length_2D" must be smaller or equal to image dimensions after resampling.')

            for x_min in range(0, (shape[0] - self.length[0] + self.stride[0]), self.stride[0]):
                if x_min + self.length[0] > shape[0]:
                    x_min = (shape[0] - self.length[0])
                x_max = x_min + self.length[0]
                for y_min in range(0, (shape[1] - self.length[1] + self.stride[1]), self.stride[1]):
                    if y_min + self.length[1] > shape[1]:
                        y_min = (shape[1] - self.length[1])
                    y_max = y_min + self.length[1]

                    # Extract patch from handlers for patch filter
                    patch = {'input': list(np.asarray(input_img)[:, x_min:x_max, y_min:y_max]),
                             'gt': list(np.asarray(gt_img)[:, x_min:x_max, y_min:y_max]) \
                                   if gt_img else [],
                             'input_metadata': input_metadata,
                             'gt_metadata': gt_metadata}
                    if self.patch_filter_fn and not self.patch_filter_fn(patch):
                        continue

                    self.indexes.append({
                        'x_min': x_min,
                        'x_max': x_max,
                        'y_min': y_min,
                        'y_max': y_max,
                        'handler_index': i})

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific processed data corresponding to index (input, ground truth, roi and metadata).

        Args:
            index (int): Slice index.
        """

        # copy.deepcopy is used to have different coordinates for reconstruction for a given handler with patch,
        # to allow a different rater at each iteration of training, and to clean transforms params from previous
        # transforms i.e. remove params from previous iterations so that the coming transforms are different
        if self.is_2d_patch:
            coord = self.indexes[index]
            if self.disk_cache:
                with self.handlers[coord['handler_index']].open(mode="rb") as f:
                    seg_pair_slice, roi_pair_slice = pickle.load(f)
            else:
                seg_pair_slice, roi_pair_slice = copy.deepcopy(self.handlers[coord['handler_index']])
        else:
            if self.disk_cache:
                with self.indexes[index].open(mode="rb") as f:
                    seg_pair_slice, roi_pair_slice = pickle.load(f)
            else:
                seg_pair_slice, roi_pair_slice = copy.deepcopy(self.indexes[index])

        # In case multiple raters
        if seg_pair_slice['gt'] and isinstance(seg_pair_slice['gt'][0], list):
            # Randomly pick a rater
            idx_rater = random.randint(0, len(seg_pair_slice['gt'][0]) - 1)
            # Use it as ground truth for this iteration
            # Note: in case of multi-class: the same rater is used across classes
            for idx_class in range(len(seg_pair_slice['gt'])):
                seg_pair_slice['gt'][idx_class] = seg_pair_slice['gt'][idx_class][idx_rater]
                seg_pair_slice['gt_metadata'][idx_class] = seg_pair_slice['gt_metadata'][idx_class][idx_rater]

        metadata_input = seg_pair_slice['input_metadata'] if seg_pair_slice['input_metadata'] is not None else []
        metadata_roi = roi_pair_slice['gt_metadata'] if roi_pair_slice['gt_metadata'] is not None else []
        metadata_gt = seg_pair_slice['gt_metadata'] if seg_pair_slice['gt_metadata'] is not None else []

        if self.is_2d_patch:
            stack_roi, metadata_roi = None, None
        else:
            # Set coordinates to the slices full size
            coord = {}
            coord['x_min'], coord['x_max'] = 0, seg_pair_slice["input"][0].shape[0]
            coord['y_min'], coord['y_max'] = 0, seg_pair_slice["input"][0].shape[1]

            # Run transforms on ROI
            # ROI goes first because params of ROICrop are needed for the followings
            stack_roi, metadata_roi = self.transform(sample=roi_pair_slice["gt"],
                                                     metadata=metadata_roi,
                                                     data_type="roi")
            # Update metadata_input with metadata_roi
            metadata_input = imed_loader_utils.update_metadata(metadata_roi, metadata_input)

        # Add coordinates of slices or patches to input metadata
        for metadata in metadata_input:
            metadata['coord'] = [coord["x_min"], coord["x_max"],
                                 coord["y_min"], coord["y_max"]]

        # Extract image and gt slices or patches from coordinates
        stack_input = np.asarray(seg_pair_slice["input"])[:,
                                 coord['x_min']:coord['x_max'],
                                 coord['y_min']:coord['y_max']]
        if seg_pair_slice["gt"]:
            stack_gt = np.asarray(seg_pair_slice["gt"])[:,
                                  coord['x_min']:coord['x_max'],
                                  coord['y_min']:coord['y_max']]
        else:
            stack_gt = []

        # Run transforms on image slices or patches
        stack_input, metadata_input = self.transform(sample=list(stack_input),
                                                     metadata=metadata_input,
                                                     data_type="im")
        # Update metadata_gt with metadata_input
        metadata_gt = imed_loader_utils.update_metadata(metadata_input, metadata_gt)
        if self.task == "segmentation":
            # Run transforms on gt slices or patches
            stack_gt, metadata_gt = self.transform(sample=list(stack_gt),
                                                   metadata=metadata_gt,
                                                   data_type="gt")
            # Make sure stack_gt is binarized
            if stack_gt is not None and not self.soft_gt:
                stack_gt = imed_postpro.threshold_predictions(stack_gt, thr=0.5).astype(np.uint8)
        else:
            # Force no transformation on labels for classification task
            # stack_gt is a tensor of size 1x1, values: 0 or 1
            # "expand(1)" is necessary to be compatible with segmentation convention: n_labelxhxwxd
            stack_gt = torch.from_numpy(seg_pair_slice["gt"][0]).expand(1)

        data_dict = {
            'input': stack_input,
            'gt': stack_gt,
            'roi': stack_roi,
            'input_metadata': metadata_input,
            'gt_metadata': metadata_gt,
            'roi_metadata': metadata_roi
        }

        # Input-level dropout to train with missing modalities
        if self.is_input_dropout:
            data_dict = dropout_input(data_dict)

        return data_dict

    def determine_cache_need(self, item: tuple, n_slice: int):
        """
        When Cache flag is not explicitly set, determine whether to cache the data or not
        Args:
            item: an EXAMPLE, typical Tuple structure contain the main data.
            n_slice: number of slice in one file_name_pairs.

        Returns:

        """
        size_item_in_bytes = get_obj_size(item)

        optimal_ram_limit = get_system_memory() * 0.5

        # Size limit: 4GB GPU RAM, keep in mind tranform etc might take MORE!
        size_estimated_dataset_GB = (size_item_in_bytes) * len(self.filename_pairs) * n_slice / 1024 ** 3
        if size_estimated_dataset_GB > optimal_ram_limit:
            logger.info(f"Estimated 2D dataset size is {size_estimated_dataset_GB} GB, which is larger than {optimal_ram_limit} GB. Auto "
                        f"enabling disk cache.")
            self.disk_cache = True
        else:
            logger.info(f"Estimated 2D dataset size is {size_estimated_dataset_GB} GB, which is smaller than {optimal_ram_limit} GB. File "
                        f"cache will not be used")
            self.disk_cache = False

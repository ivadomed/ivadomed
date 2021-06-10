import copy
import random
import nibabel as nib
import numpy as np
import torch
import pandas as pd
import os
import imageio
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, adaptative as imed_adaptative, film as imed_film
from ivadomed.object_detection import utils as imed_obj_detect


def load_dataset(bids_df, data_list, transforms_params, model_params, target_suffix, roi_params,
                 contrast_params, slice_filter_params, slice_axis, multichannel,
                 dataset_type="training", requires_undo=False, metadata_type=None,
                 object_detection_params=None, soft_gt=False, device=None,
                 cuda_available=None, is_input_dropout=False, **kwargs):
    """Get loader appropriate loader according to model type. Available loaders are Bids3DDataset for 3D data,
    BidsDataset for 2D data and HDF5Dataset for HeMIS.

    Args:
        bids_df (BidsDataframe): Object containing dataframe with all BIDS image files and their metadata.
        data_list (list): Subject names list.
        transforms_params (dict): Dictionary containing transformations for "training", "validation", "testing" (keys),
            eg output of imed_transforms.get_subdatasets_transforms.
        model_params (dict): Dictionary containing model parameters.
        target_suffix (list of str): List of suffixes for target masks.
        roi_params (dict): Contains ROI related parameters.
        contrast_params (dict): Contains image contrasts related parameters.
        slice_filter_params (dict): Contains slice_filter parameters, see :doc:`configuration_file` for more details.
        slice_axis (string): Choice between "axial", "sagittal", "coronal" ; controls the axis used to extract the 2D
            data from 3D NifTI files. 2D PNG/TIF/JPG files use default "axial.
        multichannel (bool): If True, the input contrasts are combined as input channels for the model. Otherwise, each
            contrast is processed individually (ie different sample / tensor).
        metadata_type (str): Choice between None, "mri_params", "contrasts".
        dataset_type (str): Choice between "training", "validation" or "testing".
        requires_undo (bool): If True, the transformations without undo_transform will be discarded.
        object_detection_params (dict): Object dection parameters.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
        truths are thresholded (0.5) after the data augmentation operations.
        is_input_dropout (bool): Return input with missing modalities.

    Returns:
        BidsDataset

    Note: For more details on the parameters transform_params, target_suffix, roi_params, contrast_params,
    slice_filter_params and object_detection_params see :doc:`configuration_file`.
    """
    # Compose transforms
    tranform_lst, _ = imed_transforms.prepare_transforms(copy.deepcopy(transforms_params), requires_undo)

    # If ROICrop is not part of the transforms, then enforce no slice filtering based on ROI data.
    if 'ROICrop' not in transforms_params:
        roi_params["slice_filter_roi"] = None

    if model_params["name"] == "Modified3DUNet" or ('is_2d' in model_params and not model_params['is_2d']):
        dataset = Bids3DDataset(bids_df=bids_df,
                                subject_file_lst=data_list,
                                target_suffix=target_suffix,
                                roi_params=roi_params,
                                contrast_params=contrast_params,
                                metadata_choice=metadata_type,
                                slice_axis=imed_utils.AXIS_DCT[slice_axis],
                                transform=tranform_lst,
                                multichannel=multichannel,
                                model_params=model_params,
                                object_detection_params=object_detection_params,
                                soft_gt=soft_gt,
                                is_input_dropout=is_input_dropout)

    elif model_params["name"] == "HeMISUnet":
        dataset = imed_adaptative.HDF5Dataset(bids_df=bids_df,
                                              subject_file_lst=data_list,
                                              model_params=model_params,
                                              contrast_params=contrast_params,
                                              target_suffix=target_suffix,
                                              slice_axis=imed_utils.AXIS_DCT[slice_axis],
                                              transform=tranform_lst,
                                              metadata_choice=metadata_type,
                                              slice_filter_fn=imed_loader_utils.SliceFilter(**slice_filter_params,
                                                                                            device=device,
                                                                                            cuda_available=cuda_available),
                                              roi_params=roi_params,
                                              object_detection_params=object_detection_params,
                                              soft_gt=soft_gt)
    else:
        # Task selection
        task = imed_utils.get_task(model_params["name"])

        dataset = BidsDataset(bids_df=bids_df,
                              subject_file_lst=data_list,
                              target_suffix=target_suffix,
                              roi_params=roi_params,
                              contrast_params=contrast_params,
                              model_params=model_params,
                              metadata_choice=metadata_type,
                              slice_axis=imed_utils.AXIS_DCT[slice_axis],
                              transform=tranform_lst,
                              multichannel=multichannel,
                              slice_filter_fn=imed_loader_utils.SliceFilter(**slice_filter_params, device=device,
                                                                            cuda_available=cuda_available),
                              soft_gt=soft_gt,
                              object_detection_params=object_detection_params,
                              task=task,
                              is_input_dropout=is_input_dropout)
        dataset.load_filenames()

    if model_params["name"] == "Modified3DUNet":
        logger.info("Loaded {} volumes of shape {} for the {} set.".format(len(dataset), dataset.length, dataset_type))
    elif model_params["name"] != "HeMISUnet" and dataset.length:
        logger.info("Loaded {} {} patches of shape {} for the {} set.".format(len(dataset), slice_axis, dataset.length,
                                                                              dataset_type))
    else:
        logger.info("Loaded {} {} slices for the {} set.".format(len(dataset), slice_axis, dataset_type))

    return dataset


def dropout_input(seg_pair):
    """Applies input-level dropout: zero to all channels minus one will be randomly set to zeros. This function verifies
    if some channels are already empty. Always at least one input channel will be kept.

    Args:
        seg_pair (dict): Batch containing torch tensors (input and gt) and metadata.

    Return:
        seg_pair (dict): Batch containing torch tensors (input and gt) and metadata with channel(s) dropped.
    """
    n_channels = seg_pair['input'].size(0)
    # Verify if the input is multichannel
    if n_channels > 1:
        # Verify if some channels are already empty
        n_unique_values = [len(torch.unique(input_data)) > 1 for input_data in seg_pair['input']]
        idx_empty = np.where(np.invert(n_unique_values))[0]

        # Select how many channels will be dropped between 0 and n_channels - 1 (keep at least one input)
        n_dropped = random.randint(0, n_channels - 1)

        if n_dropped > len(idx_empty):
            # Remove empty channel to the number of channels to drop
            n_dropped = n_dropped - len(idx_empty)
            # Select which channels will be dropped
            idx_dropped = []
            while len(idx_dropped) != n_dropped:
                idx = random.randint(0, n_channels - 1)
                # Don't include the empty channel in the dropped channels
                if idx not in idx_empty:
                    idx_dropped.append(idx)
        else:
            idx_dropped = idx_empty

        seg_pair['input'][idx_dropped] = torch.zeros_like(seg_pair['input'][idx_dropped])

    else:
        logger.warning("\n Impossible to apply input-level dropout since input is not multi-channel.")

    return seg_pair


class SegmentationPair(object):
    """This class is used to build segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).

    Args:
        input_filenames (list of str): The input filename list (supported by nibabel). For single channel, the list will
            contain 1 input filename.
        gt_filenames (list of str): The ground-truth filenames list.
        metadata (list): Metadata list with each item corresponding to an image (contrast) in input_filenames.
            For single channel, the list will contain metadata related to one image.
        cache (bool): If the data should be cached in memory or not.
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        prepro_transforms (dict): Output of get_preprocessing_transforms.

    Attributes:
        input_filenames (list): List of input filenames.
        gt_filenames (list): List of ground truth filenames.
        metadata (dict): Dictionary containing metadata of input and gt.
        cache (bool): If the data should be cached in memory or not.
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        prepro_transforms (dict): Transforms to be applied before training.
        input_handle (list): List of input NifTI data as 'nibabel.nifti1.Nifti1Image' object
        gt_handle (list): List of gt (ground truth) NifTI data as 'nibabel.nifti1.Nifti1Image' object
        extension (str): File extension of input files
    """

    def __init__(self, input_filenames, gt_filenames, metadata=None, slice_axis=2, cache=True, prepro_transforms=None,
                 soft_gt=False):

        self.input_filenames = input_filenames
        self.gt_filenames = gt_filenames
        self.metadata = metadata
        self.cache = cache
        self.slice_axis = slice_axis
        self.soft_gt = soft_gt
        self.prepro_transforms = prepro_transforms
        # list of the images
        self.input_handle = []

        # Ordered list of supported file extensions
        # TODO: Implement support of the following OMETIFF formats (#739):
        # [".ome.tif", ".ome.tiff", ".ome.tf2", ".ome.tf8", ".ome.btf"]
        # They are included in the list to avoid a ".ome.tif" or ".ome.tiff" following the ".tif" or ".tiff" pipeline
        ext_lst = [".nii", ".nii.gz", ".ome.tif", ".ome.tiff", ".ome.tf2", ".ome.tf8", ".ome.btf", ".tif",
                   ".tiff", ".png", ".jpg", ".jpeg"]

        # Returns the first match from the list
        self.extension = next((ext for ext in ext_lst if input_filenames[0].lower().endswith(ext)), None)
        # TODO: remove "ome" from condition when implementing OMETIFF support (#739)
        if (not self.extension) or ("ome" in self.extension):
            raise RuntimeError("The input file type of '{}' is not supported".format(input_filenames[0]))

        # loop over the filenames (list)
        for input_file in self.input_filenames:
            input_img = self.read_file(input_file)
            self.input_handle.append(input_img)
            if len(input_img.shape) > 3:
                raise RuntimeError("4-dimensional volumes not supported.")

        # list of GT for multiclass segmentation
        self.gt_handle = []

        # Labeled data (ie not inference time)
        if self.gt_filenames is not None:
            if not isinstance(self.gt_filenames, list):
                self.gt_filenames = [self.gt_filenames]
            for gt in self.gt_filenames:
                if gt is not None:
                    if isinstance(gt, str):  # this tissue has annotation from only one rater
                        self.gt_handle.append(self.read_file(gt))
                    else:  # this tissue has annotation from several raters
                        self.gt_handle.append([self.read_file(gt_rater) for gt_rater in gt])
                else:
                    self.gt_handle.append(None)

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_filenames is not None and self.gt_filenames[0] is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        for idx, handle in enumerate(self.input_handle):
            self.input_handle[idx] = nib.as_closest_canonical(handle)

        # Labeled data (ie not inference time)
        if self.gt_filenames is not None:
            for idx, gt in enumerate(self.gt_handle):
                if gt is not None:
                    if not isinstance(gt, list):  # this tissue has annotation from only one rater
                        self.gt_handle[idx] = nib.as_closest_canonical(gt)
                    else:  # this tissue has annotation from several raters
                        self.gt_handle[idx] = [nib.as_closest_canonical(gt_rater) for gt_rater in gt]

        # If binary classification, then extract labels from GT mask

        if self.metadata:
            self.metadata = []
            for data, input_filename in zip(metadata, input_filenames):
                data["input_filenames"] = input_filename
                data["gt_filenames"] = gt_filenames
                self.metadata.append(data)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input and ground truth shapes."""
        input_shape = []
        for handle in self.input_handle:
            shape = imed_loader_utils.orient_shapes_hwd(handle.header.get_data_shape(), self.slice_axis)
            input_shape.append(tuple(shape))

            if not len(set(input_shape)):
                raise RuntimeError('Inputs have different dimensions.')

        gt_shape = []

        for gt in self.gt_handle:
            if gt is not None:
                if not isinstance(gt, list):  # this tissue has annotation from only one rater
                    gt = [gt]
                for gt_rater in gt:
                    shape = imed_loader_utils.orient_shapes_hwd(gt_rater.header.get_data_shape(), self.slice_axis)
                    gt_shape.append(tuple(shape))

                if not len(set(gt_shape)):
                    raise RuntimeError('Labels have different dimensions.')

        return input_shape[0], gt_shape[0] if len(gt_shape) else None

    def get_pair_data(self):
        """Return the tuple (input, ground truth) with the data content in numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'

        input_data = []
        for handle in self.input_handle:
            hwd_oriented = imed_loader_utils.orient_img_hwd(handle.get_fdata(cache_mode, dtype=np.float32), self.slice_axis)
            input_data.append(hwd_oriented)

        gt_data = []
        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        for gt in self.gt_handle:
            if gt is not None:
                if not isinstance(gt, list):  # this tissue has annotation from only one rater
                    hwd_oriented = imed_loader_utils.orient_img_hwd(gt.get_fdata(cache_mode, dtype=np.float32), self.slice_axis)
                    gt_data.append(hwd_oriented)
                else:  # this tissue has annotation from several raters
                    hwd_oriented_list = [
                        imed_loader_utils.orient_img_hwd(gt_rater.get_fdata(cache_mode, dtype=np.float32),
                                                         self.slice_axis) for gt_rater in gt]
                    gt_data.append([hwd_oriented.astype(data_type) for hwd_oriented in hwd_oriented_list])
            else:
                gt_data.append(
                    np.zeros(imed_loader_utils.orient_shapes_hwd(self.input_handle[0].shape, self.slice_axis),
                             dtype=np.float32).astype(np.uint8))

        return input_data, gt_data

    def get_pair_metadata(self, slice_index=0, coord=None):
        """Return dictionary containing input and gt metadata.

        Args:
            slice_index (int): Index of 2D slice if 2D model is used, else 0.
            coord (tuple or list): Coordinates of subvolume in volume if 3D model is used, else None.

        Returns:
            dict: Input and gt metadata.
        """
        gt_meta_dict = []
        for idx_class, gt in enumerate(self.gt_handle):
            if gt is not None:
                if not isinstance(gt, list):  # this tissue has annotation from only one rater
                    gt_meta_dict.append(imed_loader_utils.SampleMetadata({
                        "zooms": imed_loader_utils.orient_shapes_hwd(gt.header.get_zooms(), self.slice_axis),
                        "data_shape": imed_loader_utils.orient_shapes_hwd(gt.header.get_data_shape(), self.slice_axis),
                        "gt_filenames": self.metadata[0]["gt_filenames"],
                        "bounding_box": self.metadata[0]["bounding_box"] if 'bounding_box' in self.metadata[
                            0] else None,
                        "data_type": 'gt',
                        "crop_params": {}
                    }))
                else:
                    gt_meta_dict.append([imed_loader_utils.SampleMetadata({
                        "zooms": imed_loader_utils.orient_shapes_hwd(gt_rater.header.get_zooms(), self.slice_axis),
                        "data_shape": imed_loader_utils.orient_shapes_hwd(gt_rater.header.get_data_shape(), self.slice_axis),
                        "gt_filenames": self.metadata[0]["gt_filenames"][idx_class][idx_rater],
                        "bounding_box": self.metadata[0]["bounding_box"] if 'bounding_box' in self.metadata[
                            0] else None,
                        "data_type": 'gt',
                        "crop_params": {}
                    }) for idx_rater, gt_rater in enumerate(gt)])

            else:
                # Temporarily append null metadata to null gt
                gt_meta_dict.append(None)

        # Replace null metadata with metadata from other existing classes of the same subject
        for idx, gt_metadata in enumerate(gt_meta_dict):
            if gt_metadata is None:
                gt_meta_dict[idx] = list(filter(None, gt_meta_dict))[0]

        input_meta_dict = []
        for handle in self.input_handle:
            input_meta_dict.append(imed_loader_utils.SampleMetadata({
                "zooms": imed_loader_utils.orient_shapes_hwd(handle.header.get_zooms(), self.slice_axis),
                "data_shape": imed_loader_utils.orient_shapes_hwd(handle.header.get_data_shape(), self.slice_axis),
                "data_type": 'im',
                "crop_params": {}
            }))

        dreturn = {
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }

        for idx, metadata in enumerate(self.metadata):  # loop across channels
            metadata["slice_index"] = slice_index
            metadata["coord"] = coord
            self.metadata[idx] = metadata
            for metadata_key in metadata.keys():  # loop across input metadata
                dreturn["input_metadata"][idx][metadata_key] = metadata[metadata_key]

        return dreturn

    def get_pair_slice(self, slice_index, gt_type="segmentation"):
        """Return the specified slice from (input, ground truth).

        Args:
            slice_index (int): Slice number.
            gt_type (str): Choice between segmentation or classification, returns mask (array) or label (int) resp.
                for the ground truth.
        """

        metadata = self.get_pair_metadata(slice_index)
        input_dataobj, gt_dataobj = self.get_pair_data()

        if self.slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        input_slices = []
        # Loop over contrasts
        for data_object in input_dataobj:
            input_slices.append(np.asarray(data_object[..., slice_index],
                                           dtype=np.float32))

        # Handle the case for unlabeled data
        if self.gt_handle is None:
            gt_slices = None
        else:
            gt_slices = []
            for gt_obj in gt_dataobj:
                if gt_type == "segmentation":
                    if not isinstance(gt_obj, list):  # annotation from only one rater
                        gt_slices.append(np.asarray(gt_obj[..., slice_index],
                                                    dtype=np.float32))
                    else:  # annotations from several raters
                        gt_slices.append([np.asarray(gt_obj_rater[..., slice_index],
                                                     dtype=np.float32) for gt_obj_rater in gt_obj])
                else:
                    if not isinstance(gt_obj, list):  # annotation from only one rater
                        gt_slices.append(np.asarray(int(np.any(gt_obj[..., slice_index]))))
                    else:  # annotations from several raters
                        gt_slices.append([np.asarray(int(np.any(gt_obj_rater[..., slice_index])))
                                          for gt_obj_rater in gt_obj])
        dreturn = {
            "input": input_slices,
            "gt": gt_slices,
            "input_metadata": metadata["input_metadata"],
            "gt_metadata": metadata["gt_metadata"],
        }

        return dreturn

    def read_file(self, filename):
        """Read file according to file extension and returns 'nibabel.nifti1.Nifti1Image' object.
        If different from NifTI file, convert to 'nibabel.nifti1.Nifti1Image' object, set zooms in object
        with PixelSize from metadata and write NifTI file.

        Args:
            filename (str): Subject filename.

        Returns:
            'nibabel.nifti1.Nifti1Image' object
        """
        if "nii" in self.extension:
            # For '.nii' and '.nii.gz' extentions
            return nib.load(filename)

        # TODO: (#739) implement OMETIFF behavior (elif "ome" in self.extension)

        else:
            # For '.png', '.tif', '.tiff', '.jpg' and 'jpeg' extentions
            # Read image as numpy array (behavior for grayscale only, behavior TBD for RGB or RBGA)
            if "tif" in self.extension:
                img = np.expand_dims(imageio.imread(filename, format='tiff-pil', as_gray=True), axis=-1)
            else:
                img = np.expand_dims(imageio.imread(filename, as_gray=True), axis=-1)

            # Convert numpy array to Nifti1Image object with 4x4 identity affine matrix
            img = nib.Nifti1Image(img, affine=np.eye(4))

            # Get pixel size in um from json metadata and convert to mm
            array_length = [2, 3]        # Accepted array length for 'PixelSize' metadata
            conversion_factor = 0.001    # Conversion factor from um to mm
            if 'PixelSize' in self.metadata[0]:
                ps_in_um = self.metadata[0]['PixelSize']
                if isinstance(ps_in_um, list) and (len(ps_in_um) in array_length):
                    ps_in_um = np.asarray(ps_in_um)
                    ps_in_um.resize(3)
                    ps_in_um[np.where(ps_in_um == 0)] = ps_in_um[0]
                elif isinstance(ps_in_um, float):
                    ps_in_um = np.asarray([ps_in_um, ps_in_um, ps_in_um])
                else:
                    raise RuntimeError("'PixelSize' metadata type is not supported. Format must be 2D [X, Y] array,"
                                       " 3D [X, Y, Z] array or float.")
                ps_in_mm = tuple(ps_in_um * conversion_factor)
            else:
                # TODO: Fix behavior for run_segment_command and inference, no BIDS metadata (#306)
                raise RuntimeError("'PixelSize' is missing from metadata")

            # Set "pixdim" (zooms) in Nifti1Image object header
            # Note: pixdim[1,2,3] must be non-zero in Nifti objects even if there is only one slice.
            # Above, when ps_in_um[2] (pixdim[3]) is 0, we assign the same PixelSize as ps_in_um[0] (pixdim[1])
            img.header.set_zooms((ps_in_mm))

            # if it doesn't already exist, save NifTI file in path_data alongside PNG file
            fname_out = filename.split('.')[0] + ".nii.gz"
            if not os.path.exists(fname_out):
                nib.save(img, fname_out)

            return img


class MRI2DSegmentationDataset(Dataset):
    """Generic class for 2D (slice-wise) segmentation dataset.

    Args:
        filename_pairs (list): a list of tuples in the format (input filename list containing all modalities,ground \
            truth filename, ROI filename, metadata).
        length (list): Size of each dimensions of the patches, length equals 0 (no patching) or 2 (2d patching).
        stride (list): Size of the pixels' shift between patches, length equals 0 (no patching) or 2 (2d patching).
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        cache (bool): if the data should be cached in memory or not.
        transform (torchvision.Compose): transformations to apply.
        slice_filter_fn (dict): Slice filter parameters, see :doc:`configuration_file` for more details.
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
        cache (bool): Tf the data should be cached in memory or not.
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        slice_filter_fn (dict): Slice filter parameters, see :doc:`configuration_file` for more details.
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

    """

    def __init__(self, filename_pairs, length=[], stride=[], slice_axis=2, cache=True, transform=None,
                 slice_filter_fn=None, task="segmentation", roi_params=None, soft_gt=False, is_input_dropout=False):
        self.indexes = []
        self.handlers = []
        self.filename_pairs = filename_pairs
        self.length = length
        self.stride = stride
        self.is_2d_patch = True if self.length else False
        self.prepro_transforms, self.transform = transform
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.n_contrasts = len(self.filename_pairs[0][0])
        if roi_params is None:
            roi_params = {"suffix": None, "slice_filter_roi": None}
        self.roi_thr = roi_params["slice_filter_roi"]
        self.slice_filter_roi = roi_params["suffix"] is not None and isinstance(self.roi_thr, int)
        self.soft_gt = soft_gt
        self.has_bounding_box = True
        self.task = task
        self.is_input_dropout = is_input_dropout


    def load_filenames(self):
        """Load preprocessed pair data (input and gt) in handler."""
        for input_filenames, gt_filenames, roi_filename, metadata in self.filename_pairs:
            roi_pair = SegmentationPair(input_filenames, roi_filename, metadata=metadata, slice_axis=self.slice_axis,
                                        cache=self.cache, prepro_transforms=self.prepro_transforms)

            seg_pair = SegmentationPair(input_filenames, gt_filenames, metadata=metadata, slice_axis=self.slice_axis,
                                        cache=self.cache, prepro_transforms=self.prepro_transforms,
                                        soft_gt=self.soft_gt)

            input_data_shape, _ = seg_pair.get_pair_shapes()

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

                item = imed_transforms.apply_preprocessing_transforms(self.prepro_transforms,
                                                                      slice_seg_pair,
                                                                      slice_roi_pair)

                # If is_2d_patch, create handlers list for indexing patch
                if self.is_2d_patch:
                    for metadata in item[0]['input_metadata']:
                        metadata['index_shape'] = item[0]['input'][0].shape
                    self.handlers.append((item))
                # else, append the whole slice to self.indexes
                else:
                    self.indexes.append(item)

        # If is_2d_patch, prepare indices of patches
        if self.is_2d_patch:
            self.prepare_indices()

    def prepare_indices(self):
        """Stores coordinates of 2d patches for training."""
        for i in range(0, len(self.handlers)):

            input_img = self.handlers[i][0]['input']
            shape = input_img[0].shape

            if len(self.length) != 2 or len(self.stride) != 2:
                raise RuntimeError('"length_2D" and "stride_2D" must be of length 2.')
            for length, stride, size in zip(self.length, self.stride, shape):
                if stride > length or stride <= 0:
                    raise RuntimeError('"stride_2D" must be greater than 0 and smaller or equal to "length_2D".')
                if length > size:
                    raise RuntimeError('"length_2D" must be smaller or equal to image dimensions.')

            for x in range(0, (shape[0] - self.length[0] + self.stride[0]), self.stride[0]):
                if x + self.length[0] > shape[0]:
                    x = (shape[0] - self.length[0])
                for y in range(0, (shape[1] - self.length[1] + self.stride[1]), self.stride[1]):
                    if y + self.length[1] > shape[1]:
                        y = (shape[1] - self.length[1])
                    self.indexes.append({
                        'x_min': x,
                        'x_max': x + self.length[0],
                        'y_min': y,
                        'y_max': y + self.length[1],
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

        if self.is_2d_patch:
            coord = self.indexes[index]
            seg_pair_slice, roi_pair_slice = self.handlers[coord['handler_index']]
        else:
            seg_pair_slice, roi_pair_slice = self.indexes[index]

        # In case multiple raters
        if seg_pair_slice['gt'] is not None and isinstance(seg_pair_slice['gt'][0], list):
            # Randomly pick a rater
            idx_rater = random.randint(0, len(seg_pair_slice['gt'][0]) - 1)
            # Use it as ground truth for this iteration
            # Note: in case of multi-class: the same rater is used across classes
            for idx_class in range(len(seg_pair_slice['gt'])):
                seg_pair_slice['gt'][idx_class] = seg_pair_slice['gt'][idx_class][idx_rater]
                seg_pair_slice['gt_metadata'][idx_class] = seg_pair_slice['gt_metadata'][idx_class][idx_rater]

        # Clean transforms params from previous transforms
        # i.e. remove params from previous iterations so that the coming transforms are different
        # Use copy to have different coordinates for reconstruction for a given handler with patch
        metadata_input = imed_loader_utils.clean_metadata(copy.deepcopy(seg_pair_slice['input_metadata']))
        metadata_roi = imed_loader_utils.clean_metadata(copy.deepcopy(roi_pair_slice['gt_metadata']))
        metadata_gt = imed_loader_utils.clean_metadata(copy.deepcopy(seg_pair_slice['gt_metadata']))

        # Run transforms on ROI
        # ROI goes first because params of ROICrop are needed for the followings
        stack_roi, metadata_roi = self.transform(sample=roi_pair_slice["gt"],
                                                 metadata=metadata_roi,
                                                 data_type="roi")

        # Update metadata_input with metadata_roi
        metadata_input = imed_loader_utils.update_metadata(metadata_roi, metadata_input)

        # Run transforms on images
        stack_input, metadata_input = self.transform(sample=seg_pair_slice["input"],
                                                     metadata=metadata_input,
                                                     data_type="im")

        # Update metadata_input with metadata_roi
        metadata_gt = imed_loader_utils.update_metadata(metadata_input, metadata_gt)

        if self.task == "segmentation":
            # Run transforms on images
            stack_gt, metadata_gt = self.transform(sample=seg_pair_slice["gt"],
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

        # If is_2d_patch, add coordinates to metadata to reconstruct image
        if self.is_2d_patch:
            shape_x = coord["x_max"] - coord["x_min"]
            shape_y = coord["y_max"] - coord["y_min"]

            for metadata in metadata_input:
                metadata['coord'] = [coord["x_min"], coord["x_max"], coord["y_min"], coord["y_max"]]

            data_dict = {
                'input': torch.zeros(stack_input.shape[0], shape_x, shape_y),
                'gt': torch.zeros(stack_gt.shape[0], shape_x, shape_y) if stack_gt is not None else None,
                'roi': torch.zeros(stack_roi.shape[0], shape_x, shape_y) if stack_roi is not None else None,
                'input_metadata': metadata_input,
                'gt_metadata': metadata_gt,
                'roi_metadata': metadata_roi
            }

            for _ in range(len(stack_input)):
                data_dict['input'] = stack_input[:,
                                      coord['x_min']:coord['x_max'],
                                      coord['y_min']:coord['y_max']]

            if stack_gt is not None:
                for _ in range(len(stack_gt)):
                    data_dict['gt'] = stack_gt[:,
                                       coord['x_min']:coord['x_max'],
                                       coord['y_min']:coord['y_max']]

            if stack_roi is not None:
                for _ in range(len(stack_roi)):
                    data_dict['roi'] = stack_roi[:,
                                       coord['x_min']:coord['x_max'],
                                       coord['y_min']:coord['y_max']]

        else:
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
                'input_metadata': metadata['input_metadata'],
                'gt_metadata': metadata['gt_metadata']
            }

            self.has_bounding_box = imed_obj_detect.verify_metadata(seg_pair, self.has_bounding_box)
            if self.has_bounding_box:
                self.prepro_transforms = imed_obj_detect.adjust_transforms(self.prepro_transforms, seg_pair,
                                                                           length=self.length,
                                                                           stride=self.stride)
            seg_pair, roi_pair = imed_transforms.apply_preprocessing_transforms(self.prepro_transforms,
                                                                                seg_pair=seg_pair)

            for metadata in seg_pair['input_metadata']:
                metadata['index_shape'] = seg_pair['input'][0].shape
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
        coord = self.indexes[index]
        seg_pair, _ = self.handlers[coord['handler_index']]

        # In case multiple raters
        if seg_pair['gt'] is not None and isinstance(seg_pair['gt'][0], list):
            # Randomly pick a rater
            idx_rater = random.randint(0, len(seg_pair['gt'][0]) - 1)
            # Use it as ground truth for this iteration
            # Note: in case of multi-class: the same rater is used across classes
            for idx_class in range(len(seg_pair['gt'])):
                seg_pair['gt'][idx_class] = seg_pair['gt'][idx_class][idx_rater]
                seg_pair['gt_metadata'][idx_class] = seg_pair['gt_metadata'][idx_class][idx_rater]

        # Clean transforms params from previous transforms
        # i.e. remove params from previous iterations so that the coming transforms are different
        # Use copy to have different coordinates for reconstruction for a given handler
        metadata_input = imed_loader_utils.clean_metadata(copy.deepcopy(seg_pair['input_metadata']))
        metadata_gt = imed_loader_utils.clean_metadata(copy.deepcopy(seg_pair['gt_metadata']))

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
            metadata['coord'] = [coord["x_min"], coord["x_max"], coord["y_min"], coord["y_max"], coord["z_min"],
                                 coord["z_max"]]

        subvolumes = {
            'input': torch.zeros(stack_input.shape[0], shape_x, shape_y, shape_z),
            'gt': torch.zeros(stack_gt.shape[0], shape_x, shape_y, shape_z) if stack_gt is not None else None,
            'input_metadata': metadata_input,
            'gt_metadata': metadata_gt
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


class Bids3DDataset(MRI3DSubVolumeSegmentationDataset):
    """BIDS specific dataset loader for 3D dataset.

    Args:
        bids_df (BidsDataframe): Object containing dataframe with all BIDS image files and their metadata.
        subject_file_lst (list): Subject filenames list.
        target_suffix (list): List of suffixes for target masks.
        model_params (dict): Dictionary containing model parameters.
        contrast_params (dict): Contains image contrasts related parameters.
        slice_axis (int): Indicates the axis used to extract slices: "axial": 2, "sagittal": 0, "coronal": 1.
        cache (bool): If the data should be cached in memory or not.
        transform (list): Transformation list (length 2) composed of preprocessing transforms (Compose) and transforms
            to apply during training (Compose).
        metadata_choice: Choice between "mri_params", "contrasts", None or False, related to FiLM.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.
        multichannel (bool): If True, the input contrasts are combined as input channels for the model. Otherwise, each
            contrast is processed individually (ie different sample / tensor).
        object_detection_params (dict): Object dection parameters.
        is_input_dropout (bool): Return input with missing modalities.
    """

    def __init__(self, bids_df, subject_file_lst, target_suffix, model_params, contrast_params, slice_axis=2,
                 cache=True, transform=None, metadata_choice=False, roi_params=None,
                 multichannel=False, object_detection_params=None, task="segmentation", soft_gt=False,
                 is_input_dropout=False):
        dataset = BidsDataset(bids_df=bids_df,
                              subject_file_lst=subject_file_lst,
                              target_suffix=target_suffix,
                              roi_params=roi_params,
                              contrast_params=contrast_params,
                              model_params=model_params,
                              metadata_choice=metadata_choice,
                              slice_axis=slice_axis,
                              transform=transform,
                              multichannel=multichannel,
                              object_detection_params=object_detection_params,
                              is_input_dropout=is_input_dropout)

        super().__init__(dataset.filename_pairs, length=model_params["length_3D"], stride=model_params["stride_3D"],
                         transform=transform, slice_axis=slice_axis, task=task, soft_gt=soft_gt,
                         is_input_dropout=is_input_dropout)


class BidsDataset(MRI2DSegmentationDataset):
    """ BIDS specific dataset loader.

    Args:
        bids_df (BidsDataframe): Object containing dataframe with all BIDS image files and their metadata.
        subject_file_lst (list): Subject filenames list.
        target_suffix (list): List of suffixes for target masks.
        contrast_params (dict): Contains image contrasts related parameters.
        model_params (dict): Dictionary containing model parameters.
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        cache (bool): If the data should be cached in memory or not.
        transform (list): Transformation list (length 2) composed of preprocessing transforms (Compose) and transforms
            to apply during training (Compose).
        metadata_choice (str): Choice between "mri_params", "contrasts", the name of a column from the
            participants.tsv file, None or False, related to FiLM.
        slice_filter_fn (SliceFilter): Class that filters slices according to their content.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.
        multichannel (bool): If True, the input contrasts are combined as input channels for the model. Otherwise, each
            contrast is processed individually (ie different sample / tensor).
        object_detection_params (dict): Object dection parameters.
        task (str): Choice between segmentation or classification. If classification: GT is discrete values, \
            If segmentation: GT is binary mask.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
        truths are thresholded (0.5) after the data augmentation operations.
        is_input_dropout (bool): Return input with missing modalities.

    Attributes:
        filename_pairs (list): A list of tuples in the format (input filename list containing all modalities,ground \
            truth filename, ROI filename, metadata).
        metadata (dict): Dictionary containing FiLM metadata.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
            truths are thresholded (0.5) after the data augmentation operations.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.

    """

    def __init__(self, bids_df, subject_file_lst, target_suffix, contrast_params, model_params, slice_axis=2,
                 cache=True, transform=None, metadata_choice=False, slice_filter_fn=None, roi_params=None,
                 multichannel=False, object_detection_params=None, task="segmentation", soft_gt=False,
                 is_input_dropout=False):

        self.roi_params = roi_params if roi_params is not None else {"suffix": None, "slice_filter_roi": None}
        self.soft_gt = soft_gt
        self.filename_pairs = []
        if metadata_choice == 'mri_params':
            self.metadata = {"FlipAngle": [], "RepetitionTime": [],
                             "EchoTime": [], "Manufacturer": []}

        # Sort subject_file_lst and create a sub-dataframe from bids_df containing only subjects from subject_file_lst
        subject_file_lst = sorted(subject_file_lst)
        df_subjects = bids_df.df[bids_df.df['filename'].isin(subject_file_lst)]
        # Backward compatibility for subject_file_lst containing participant_ids instead of filenames
        if df_subjects.empty:
            df_subjects = bids_df.df[bids_df.df['participant_id'].isin(subject_file_lst)]
            subject_file_lst = sorted(df_subjects['filename'].to_list())

        # Create a dictionary with the number of subjects for each contrast of contrast_balance
        tot = {contrast: df_subjects['suffix'].str.fullmatch(contrast).value_counts()[True]
               for contrast in contrast_params["balance"].keys()}

        # Create a counter that helps to balance the contrasts
        c = {contrast: 0 for contrast in contrast_params["balance"].keys()}

        # Get a list of subject_ids for multichannel_subjects (prefix filename without modality suffix and extension)
        subject_ids = []
        for subject in subject_file_lst:
            subject_ids.append(subject.split('.')[0].split('_')[0])
        subject_ids = sorted(list(set(subject_ids)))

        # Create multichannel_subjects dictionary for each subject_id
        multichannel_subjects = {}
        idx_dict = {}
        if multichannel:
            num_contrast = len(contrast_params["contrast_lst"])
            for idx, contrast in enumerate(contrast_params["contrast_lst"]):
                idx_dict[contrast] = idx
            multichannel_subjects = {subject: {"absolute_paths": [None] * num_contrast,
                                               "deriv_path": None,
                                               "roi_filename": None,
                                               "metadata": [None] * num_contrast} for subject in subject_ids}

        # Get all subjects path from bids_df for bounding box
        get_all_subj_path = bids_df.df[bids_df.df['filename']
                                .str.contains('|'.join(bids_df.get_subject_fnames()))]['path'].to_list()

        # Load bounding box from list of path
        bounding_box_dict = imed_obj_detect.load_bounding_boxes(object_detection_params,
                                                                get_all_subj_path,
                                                                slice_axis,
                                                                contrast_params["contrast_lst"])

        # Get all derivatives filenames from bids_df
        all_deriv = bids_df.get_deriv_fnames()

        # Create filename_pairs
        for subject in tqdm(subject_file_lst, desc="Loading dataset"):
            df_sub, roi_filename, target_filename, metadata = self.create_filename_pair(multichannel_subjects, subject, 
                                                                                        c, tot, multichannel, df_subjects, 
                                                                                        contrast_params, target_suffix, 
                                                                                        all_deriv, bids_df, bounding_box_dict, 
                                                                                        idx_dict, metadata_choice)
            # Fill multichannel dictionary
            # subj_id is the filename without modality suffix and extension
            if multichannel:
                multichannel_subjects = self.fill_multichannel_dict(multichannel_subjects, subject, idx_dict, df_sub, 
                                                                    roi_filename, target_filename, metadata)
            else:
                self.filename_pairs.append(([df_sub['path'].values[0]],
                                            target_filename, roi_filename, [metadata]))

        if multichannel:
            for subject in multichannel_subjects.values():
                if None not in subject["absolute_paths"]:
                    self.filename_pairs.append((subject["absolute_paths"], subject["deriv_path"],
                                                subject["roi_filename"], subject["metadata"]))

        if not self.filename_pairs:
            raise Exception('No subjects were selected - check selection of parameters on config.json (e.g. center '
                            'selected + target_suffix)')

        length = model_params["length_2D"] if "length_2D" in model_params else []
        stride = model_params["stride_2D"] if "stride_2D" in model_params else []

        super().__init__(self.filename_pairs, length, stride, slice_axis, cache, transform, slice_filter_fn, task, self.roi_params,
                         self.soft_gt, is_input_dropout)

    def get_target_filename(self, target_suffix, target_filename, derivative):
        for idx, suffix_list in enumerate(target_suffix):
            # If suffix_list is a string, then only one rater annotation per class is available.
            # Otherwise, multiple raters segmented the same class.
            if isinstance(suffix_list, list):
                for suffix in suffix_list:
                    if suffix in derivative:
                        target_filename[idx].append(derivative)
            elif suffix_list in derivative:
                target_filename[idx] = derivative
            

    def create_metadata_dict(self, metadata_choice, df_sub, bids_df):
        # add custom data to metadata
        if metadata_choice not in df_sub.columns:
            raise ValueError("The following metadata cannot be found: {}. "
                                "Invalid metadata choice.".format(metadata_choice))
        metadata[metadata_choice] = df_sub[metadata_choice].values[0]
        # Create metadata dict for OHE
        data_lst = sorted(set(bids_df.df[metadata_choice].dropna().values))
        metadata_dict = {}
        for idx, data in enumerate(data_lst):
            metadata_dict[data] = idx
        metadata['metadata_dict'] = metadata_dict

    def fill_multichannel_dict(self, multichannel_subjects, subject, idx_dict, df_sub, roi_filename, target_filename, metadata):
        idx = idx_dict[df_sub['suffix'].values[0]]
        subj_id = subject.split('.')[0].split('_')[0]
        multichannel_subjects[subj_id]["absolute_paths"][idx] = df_sub['path'].values[0]
        multichannel_subjects[subj_id]["deriv_path"] = target_filename
        multichannel_subjects[subj_id]["metadata"][idx] = metadata
        if roi_filename:
            multichannel_subjects[subj_id]["roi_filename"] = roi_filename
        return multichannel_subjects


    def create_filename_pair(self, multichannel_subjects, subject, c, tot, multichannel, df_subjects, contrast_params, 
                            target_suffix, all_deriv, bids_df, bounding_box_dict, idx_dict, metadata_choice):
        df_sub = df_subjects.loc[df_subjects['filename'] == subject]

        # Training & Validation: do not consider the contrasts over the threshold contained in contrast_balance
        contrast = df_sub['suffix'].values[0]
        if contrast in (contrast_params["balance"].keys()):
            c[contrast] = c[contrast] + 1
            if c[contrast] / tot[contrast] > contrast_params["balance"][contrast]:
                return
        if isinstance(target_suffix[0], str):
            target_filename, roi_filename = [None] * len(target_suffix), None
        else:
            target_filename, roi_filename = [[] for _ in range(len(target_suffix))], None

        derivatives = bids_df.df[bids_df.df['filename']
                        .str.contains('|'.join(bids_df.get_derivatives(subject, all_deriv)))]['path'].to_list()

        for derivative in derivatives:
            self.get_target_filename(target_suffix, target_filename, derivative)
            if not (self.roi_params["suffix"] is None) and self.roi_params["suffix"] in derivative:
                roi_filename = [derivative]

        if (not any(target_filename)) or (not (self.roi_params["suffix"] is None) and (roi_filename is None)):
            return

        metadata = df_sub.to_dict(orient='records')[0]
        metadata['contrast'] = contrast

        if len(bounding_box_dict):
            # Take only one bounding box for cropping
            metadata['bounding_box'] = bounding_box_dict[str(df_sub['path'].values[0])][0]

        if metadata_choice == 'mri_params':
            if not all([imed_film.check_isMRIparam(m, metadata, subject, self.metadata) for m in
                        self.metadata.keys()]):
                return

        elif metadata_choice and metadata_choice != 'contrasts' and metadata_choice is not None:
            self.create_metadata_dict(metadata_choice, df_sub, bids_df)
        
        return df_sub, roi_filename, target_filename, metadata

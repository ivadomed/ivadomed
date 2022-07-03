import functools
import copy
import numpy as np
import torchio as tio
from typing import Tuple
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.transforms.compose import Compose
from ivadomed.transforms.undo_compose import UndoCompose
from ivadomed.keywords import TransformationKW, MetadataKW


def multichannel_capable(wrapped):
    """Decorator to make a given function compatible multichannel images.

    Args:
        wrapped: Given function.

    Returns:
        Functions' return.
    """

    @functools.wraps(wrapped)
    def wrapper(self, sample, metadata):
        if isinstance(sample, list):
            list_data, list_metadata = [], []
            for s_cur, m_cur in zip(sample, metadata):
                if len(list_metadata) > 0:
                    if not isinstance(list_metadata[-1], list):
                        imed_loader_utils.update_metadata([list_metadata[-1]], [m_cur])
                    else:
                        imed_loader_utils.update_metadata(list_metadata[-1], [m_cur])
                # Run function for each sample of the list
                data_cur, metadata_cur = wrapped(self, s_cur, m_cur)
                list_data.append(data_cur)
                list_metadata.append(metadata_cur)
            return list_data, list_metadata
        # If sample is None, then return a pair (None, None)
        if sample is None:
            return None, None
        else:
            return wrapped(self, sample, metadata)

    return wrapper


def two_dim_compatible(wrapped):
    """Decorator to make a given function compatible 2D or 3D images.

    Args:
        wrapped: Given function.

    Returns:
        Functions' return.
    """

    @functools.wraps(wrapped)
    def wrapper(self, sample, metadata):
        # Check if sample is 2D
        if len(sample.shape) == 2:
            # Add one dimension
            sample = np.expand_dims(sample, axis=-1)
            # Run transform
            result_sample, result_metadata = wrapped(self, sample, metadata)
            # Remove last dimension
            return np.squeeze(result_sample, axis=-1), result_metadata
        else:
            return wrapped(self, sample, metadata)

    return wrapper


def get_subdatasets_transforms(transform_params):
    """Get transformation parameters for each subdataset: training, validation and testing.

    Args:
        transform_params (dict):

    Returns:
        dict, dict, dict: Training, Validation and Testing transformations.
    """
    transform_params = copy.deepcopy(transform_params)
    train, valid, test = {}, {}, {}
    subdataset_default = ["training", "validation", "testing"]
    # Loop across transformations
    for transform_name in transform_params:
        subdataset_list = ["training", "validation", "testing"]
        # Only consider subdatasets listed in dataset_type
        if "dataset_type" in transform_params[transform_name]:
            subdataset_list = transform_params[transform_name]["dataset_type"]
        # Add current transformation to the relevant subdataset transformation dictionaries
        for subds_name, subds_dict in zip(subdataset_default, [train, valid, test]):
            if subds_name in subdataset_list:
                subds_dict[transform_name] = transform_params[transform_name]
                if "dataset_type" in subds_dict[transform_name]:
                    del subds_dict[transform_name]["dataset_type"]
    return train, valid, test


def get_preprocessing_transforms(transforms):
    """Checks the transformations parameters and selects the transformations which are done during preprocessing only.

    Args:
        transforms (dict): Transformation dictionary.

    Returns:
        dict: Preprocessing transforms.
    """
    original_transforms = copy.deepcopy(transforms)
    preprocessing_transforms = copy.deepcopy(transforms)
    for idx, tr in enumerate(original_transforms):
        if tr == TransformationKW.RESAMPLE or tr == TransformationKW.CENTERCROP or tr == TransformationKW.ROICROP:
            del transforms[tr]
        else:
            del preprocessing_transforms[tr]

    return preprocessing_transforms


def apply_preprocessing_transforms(transforms, seg_pair, roi_pair=None) -> Tuple[dict, dict]:
    """
    Applies preprocessing transforms to segmentation pair (input, gt and metadata).

    Args:
        transforms (Compose): Preprocessing transforms.
        seg_pair (dict): Segmentation pair containing input and gt.
        roi_pair (dict): Segementation pair containing input and roi.

    Returns:
        tuple: Segmentation pair and roi pair.
    """
    if transforms is None:
        return (seg_pair, roi_pair)

    metadata_input = seg_pair['input_metadata']
    if roi_pair is not None:
        stack_roi, metadata_roi = transforms(sample=roi_pair["gt"],
                                             metadata=roi_pair['gt_metadata'],
                                             data_type="roi",
                                             preprocessing=True)
        metadata_input = imed_loader_utils.update_metadata(metadata_roi, metadata_input)
    # Run transforms on images
    stack_input, metadata_input = transforms(sample=seg_pair["input"],
                                             metadata=metadata_input,
                                             data_type="im",
                                             preprocessing=True)
    # Run transforms on images
    metadata_gt = imed_loader_utils.update_metadata(metadata_input, seg_pair['gt_metadata'])
    stack_gt, metadata_gt = transforms(sample=seg_pair["gt"],
                                       metadata=metadata_gt,
                                       data_type="gt",
                                       preprocessing=True)

    seg_pair = {
        'input': stack_input,
        'gt': stack_gt,
        MetadataKW.INPUT_METADATA: metadata_input,
        MetadataKW.GT_METADATA: metadata_gt
    }

    if roi_pair is not None and len(roi_pair['gt']):
        roi_pair = {
            'input': stack_input,
            'gt': stack_roi,
            MetadataKW.INPUT_METADATA: metadata_input,
            MetadataKW.GT_METADATA: metadata_roi
        }
    return (seg_pair, roi_pair)


def prepare_transforms(transform_dict, requires_undo=True):
    """
    This function separates the preprocessing transforms from the others and generates the undo transforms related.

    Args:
        transform_dict (dict): Dictionary containing the transforms and there parameters.
        requires_undo (bool): Boolean indicating if transforms can be undone.

    Returns:
        list, UndoCompose: transform lst containing the preprocessing transforms and regular transforms, UndoCompose
            object containing the transform to undo.
    """
    training_undo_transform = None
    if requires_undo:
        training_undo_transform = UndoCompose(Compose(transform_dict.copy()))
    preprocessing_transforms = get_preprocessing_transforms(transform_dict)
    prepro_transforms = Compose(preprocessing_transforms, requires_undo=requires_undo)
    transforms = Compose(transform_dict, requires_undo=requires_undo)
    tranform_lst = [prepro_transforms if len(preprocessing_transforms) else None, transforms]
    return tranform_lst, training_undo_transform


def tio_transform(x, transform):
    """
    Applies TorchIO transformations to a given image and returns the transformed image and history.

    Args:
        x (np.ndarray): input image
        transform (tio.transforms.Transform): TorchIO transform

    Returns:
        np.ndarray, list: transformed image, history of parameters used for the applied transformation
    """
    tio_subject = tio.Subject(input=tio.ScalarImage(tensor=x[np.newaxis, ...]))
    transformed = transform(tio_subject)
    return transformed.input.numpy()[0], transformed.get_composed_history()

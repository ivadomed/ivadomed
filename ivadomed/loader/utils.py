import collections
import re

import numpy as np
import torch
from bids_neuropoly import bids
from sklearn.model_selection import train_test_split
from torch._six import string_classes, int_classes

__numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

TRANSFORM_PARAMS = ['elastic', 'rotation', 'offset', 'crop_params', 'reverse', 'affine', 'gaussian_noise']


def split_dataset(path_folder, center_test_lst, split_method, random_seed, train_frac=0.8, test_frac=0.1):
    # read participants.tsv as pandas dataframe
    df = bids.BIDS(path_folder).participants.content
    X_test = []
    X_train = []
    X_val = []
    if split_method == 'per_center':
        # make sure that subjects coming from some centers are unseen during training
        X_test = df[df['institution_id'].isin(center_test_lst)]['participant_id'].tolist()
        X_remain = df[~df['institution_id'].isin(center_test_lst)]['participant_id'].tolist()

        # split using sklearn function
        X_train, X_tmp = train_test_split(X_remain, train_size=train_frac, random_state=random_seed)
        if len(X_test):  # X_test contains data from centers unseen during the training, eg SpineGeneric
            X_val = X_tmp
        else:  # X_test contains data from centers seen during the training, eg gm_challenge
            X_val, X_test = train_test_split(X_tmp, train_size=0.5, random_state=random_seed)
    elif split_method == 'per_patient':
        # Separate dataset in test, train and validation using sklearn function
        X_train, X_remain = train_test_split(df['participant_id'].tolist(), train_size=train_frac,
                                             random_state=random_seed)
        X_test, X_val = train_test_split(X_remain, train_size=test_frac / (1 - train_frac), random_state=random_seed)

    else:
        print(" {split_method} is not a supported split method")

    return X_train, X_val, X_test


def imed_collate(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        stacked = torch.stack(batch, 0)
        return stacked
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return __numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: imed_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        return [imed_collate(samples) for samples in batch]

    return batch


def filter_roi(ds, nb_nonzero_thr):
    """Filter slices from dataset using ROI data.

    This function loops across the dataset (ds) and discards slices where the number of
    non-zero voxels within the ROI slice (e.g. centerline, SC segmentation) is inferior or
    equal to a given threshold (nb_nonzero_thr).

    Args:
        ds (mt_datasets.MRI2DSegmentationDataset): Dataset.
        nb_nonzero_thr (int): Threshold.

    Returns:
        mt_datasets.MRI2DSegmentationDataset: Dataset without filtered slices.

    """
    filter_indexes = []
    for segpair, slice_roi_pair in ds.indexes:
        roi_data = slice_roi_pair['gt']

        # Discard slices with less nonzero voxels than nb_nonzero_thr
        if not np.any(roi_data):
            continue
        if np.count_nonzero(roi_data) <= nb_nonzero_thr:
            continue

        filter_indexes.append((segpair, slice_roi_pair))

    # Update dataset
    ds.indexes = filter_indexes
    return ds


def orient_img_hwd(data, slice_axis):
    """
    Orient a given RAS image to height, width, depth according to slice axis.
    :return: numpy array oriented with the following dimensions: (height, width, depth)
    """
    if slice_axis == 0:
        return data.transpose(2, 1, 0)
    elif slice_axis == 1:
        return data.transpose(2, 0, 1)
    elif slice_axis == 2:
        return data


def orient_img_ras(data, slice_axis):
    """
    Orient a given numpy array with dimensions (height, width, depth) to RAS oriented.
    :return: numpy array oriented in RAS
    """
    if slice_axis == 0:
        return data.transpose(2, 1, 0) if len(data.shape) == 3 else data.transpose(0, 3, 2, 1)
    elif slice_axis == 1:
        return data.transpose(1, 2, 0) if len(data.shape) == 3 else data.transpose(0, 2, 3, 1)
    elif slice_axis == 2:
        return data


def orient_shapes_hwd(data, slice_axis):
    """
    Swap dimensions according to match the height, width, depth orientation
    :return: numpy array containing the swapped dimensions
    """
    if slice_axis == 0:
        return np.array(data)[[2, 1, 0]]
    elif slice_axis == 1:
        return np.array(data)[[2, 0, 1]]
    elif slice_axis == 2:
        return np.array(data)


class SampleMetadata(object):
    def __init__(self, d=None):
        self.metadata = {} or d

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def __getitem__(self, key):
        return self.metadata[key]

    def __contains__(self, key):
        return key in self.metadata

    def items(self):
        return self.metadata.items()

    def _update(self, ref, list_keys):
        for k in list_keys:
            if k not in self.metadata.keys() and k in ref.metadata.keys():
                self.metadata[k] = ref.metadata[k]

    def keys(self):
        return self.metadata.keys()


class BalancedSampler(torch.utils.data.sampler.Sampler):
    """Estimate sampling weights in order to rebalance the
    class distributions from an imbalanced dataset.
    """

    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))

        self.nb_samples = len(self.indices)

        cmpt_label = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in cmpt_label:
                cmpt_label[label] += 1
            else:
                cmpt_label[label] = 1

        weights = [1.0 / cmpt_label[self._get_label(dataset, idx)]
                   for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    @staticmethod
    def _get_label(dataset, idx):
        # For now, only supported with single label
        sample_gt = np.array(dataset[idx]['gt'][0])
        if np.any(sample_gt):
            return 1
        else:
            return 0

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.nb_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def clean_metadata(metadata_lst):
    metadata_out = []
    for metadata_cur in metadata_lst:
        for key_ in list(metadata_cur.keys()):
            if key_ in TRANSFORM_PARAMS:
                del metadata_cur.metadata[key_]
        metadata_out.append(metadata_cur)
    return metadata_out


def update_metadata(metadata_src_lst, metadata_dest_lst):
    if metadata_src_lst and metadata_dest_lst:
        if len(metadata_src_lst) > len(metadata_dest_lst):
            metadata_dest_lst = metadata_dest_lst + [SampleMetadata({})] * \
                                (len(metadata_src_lst) - len(metadata_dest_lst))
        for idx in range(len(metadata_src_lst)):
            metadata_dest_lst[idx]._update(metadata_src_lst[idx], TRANSFORM_PARAMS)
    return metadata_dest_lst

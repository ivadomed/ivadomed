import collections
import re

import numpy as np
import torch
import joblib
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

TRANSFORM_PARAMS = ['elastic', 'rotation', 'scale', 'offset', 'crop_params', 'reverse', 'translation', 'gaussian_noise']


def split_dataset(path_folder, center_test_lst, split_method, random_seed, train_frac=0.8, test_frac=0.1):
    """Splits list of subject into training, validation and testing datasets either according to their center or per
    patient. In the 'per_center' option the centers associated the subjects are split according the train, test and
    validation fraction whereas in the 'per_patient', the patients are directly separated according to these fractions.

    Args:
        path_folder (str): Path to BIDS folder.
        center_test_lst (list): list of centers to include in the testing set.
        split_method (str): Between 'per_center' or 'per_person'. If 'per_center' the separation fraction are
            applied to centers, if 'per_person' they are applied to the subject list.
        random_seed (int): Random seed to ensure reproducible splits.
        train_frac (float): Between 0 and 1. Represents the train set proportion.
        test_frac (float): Between 0 and 1. Represents the test set proportion.

    Returns:
        list, list, list: Train, validation and test subjects list.
    """
    # read participants.tsv as pandas dataframe
    df = bids.BIDS(path_folder).participants.content
    X_test = []
    X_train = []
    X_val = []
    if split_method == 'per_center':
        # make sure that subjects coming from some centers are unseen during training
        if len(center_test_lst) == 0:
            centers = list(df['institution_id'])
            test_frac = test_frac if test_frac >= 1 / len(centers) else 1 / len(centers)
            center_test_lst, _ = train_test_split(centers, train_size=test_frac, random_state=random_seed)

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


def get_new_subject_split(path_folder, center_test, split_method, random_seed,
                          train_frac, test_frac, log_directory):
    """Randomly split dataset between training / validation / testing.

    Randomly split dataset between training / validation / testing\
        and save it in log_directory + "/split_datasets.joblib".

    Args:
        path_folder (string): Dataset folder.
        center_test (list): List of centers to include in the testing set.
        split_method (string): See imed_loader_utils.split_dataset.
        random_seed (int): Random seed.
        train_frac (float): Training dataset proportion, between 0 and 1.
        test_frac (float): Testing dataset proportionm between 0 and 1.
        log_directory (string): Output folder.

    Returns:
        list, list list: Training, validation and testing subjects lists.
    """
    train_lst, valid_lst, test_lst = split_dataset(path_folder=path_folder,
                                                   center_test_lst=center_test,
                                                   split_method=split_method,
                                                   random_seed=random_seed,
                                                   train_frac=train_frac,
                                                   test_frac=test_frac)

    # save the subject distribution
    split_dct = {'train': train_lst, 'valid': valid_lst, 'test': test_lst}
    joblib.dump(split_dct, "./" + log_directory + "/split_datasets.joblib")

    return train_lst, valid_lst, test_lst


def get_subdatasets_subjects_list(split_params, bids_path, log_directory):
    """Get lists of subjects for each sub-dataset between training / validation / testing.

    Args:
        split_params (dict): Split parameters, see :doc:`configuration_file` for more details.
        bids_path (str): Path to the BIDS dataset.
        log_directory (str): Output folder.

    Returns:
        list, list list: Training, validation and testing subjects lists.
    """
    if split_params["fname_split"]:
        # Load subjects lists
        old_split = joblib.load(split_params["fname_split"])
        train_lst, valid_lst, test_lst = old_split['train'], old_split['valid'], old_split['test']
    else:
        train_lst, valid_lst, test_lst = get_new_subject_split(path_folder=bids_path,
                                                               center_test=split_params['center_test'],
                                                               split_method=split_params['method'],
                                                               random_seed=split_params['random_seed'],
                                                               train_frac=split_params['train_fraction'],
                                                               test_frac=split_params['test_fraction'],
                                                               log_directory=log_directory)
    return train_lst, valid_lst, test_lst


def imed_collate(batch):
    """Collates data to create batches

    Args:
        batch (dict): Contains input and gt data with their corresponding metadata.

    Returns:
        list or dict or str or tensor: Collated data.
    """
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


def filter_roi(roi_data, nb_nonzero_thr):
    """Filter slices from dataset using ROI data.

    This function filters slices (roi_data) where the number of non-zero voxels within the ROI slice (e.g. centerline,
    SC segmentation) is inferior or equal to a given threshold (nb_nonzero_thr).

    Args:
        roi_data (nd.array): ROI slice.
        nb_nonzero_thr (int): Threshold.

    Returns:
        bool: True if the slice needs to be filtered, False otherwise.
    """
    # Discard slices with less nonzero voxels than nb_nonzero_thr
    return not np.any(roi_data) or np.count_nonzero(roi_data) <= nb_nonzero_thr


def orient_img_hwd(data, slice_axis):
    """Orient a given RAS image to height, width, depth according to slice axis.

    Args:
        data (ndarray): RAS oriented data.
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.

    Returns:
        ndarray: Array oriented with the following dimensions: (height, width, depth).
    """
    if slice_axis == 0:
        return data.transpose(2, 1, 0)
    elif slice_axis == 1:
        return data.transpose(2, 0, 1)
    elif slice_axis == 2:
        return data


def orient_img_ras(data, slice_axis):
    """Orient a given array with dimensions (height, width, depth) to RAS orientation.

    Args:
        data (ndarray): Data with following dimensions (Height, Width, Depth).
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.

    Returns:
        ndarray: Array oriented in RAS.
    """

    if slice_axis == 0:
        return data.transpose(2, 1, 0) if len(data.shape) == 3 else data.transpose(0, 3, 2, 1)
    elif slice_axis == 1:
        return data.transpose(1, 2, 0) if len(data.shape) == 3 else data.transpose(0, 2, 3, 1)
    elif slice_axis == 2:
        return data


def orient_shapes_hwd(data, slice_axis):
    """Swap dimensions according to match the height, width, depth orientation.

    Args:
        data (list or tuple): Shape or numbers associated with each image dimension (e.i. image resolution).
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.

    Returns:
        ndarray: Reoriented vector.
    """
    if slice_axis == 0:
        return np.array(data)[[2, 1, 0]]
    elif slice_axis == 1:
        return np.array(data)[[2, 0, 1]]
    elif slice_axis == 2:
        return np.array(data)


class SampleMetadata(object):
    """Metadata class to help update, get and set metadata values.

    Args:
        d (dict): Initial metadata.

    Attributes:
        metadata (dict): Image metadata.
    """
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
        """Update metadata keys with a reference metadata. A given list of metadata keys will be changed and given the
        values of the reference metadata.

        Args:
            ref (SampleMetadata): Reference metadata object.
            list_keys (list): List of keys that need to be updated.
        """
        for k in list_keys:
            if (k not in self.metadata.keys() or not bool(self.metadata[k])) and k in ref.metadata.keys():
                self.metadata[k] = ref.metadata[k]

    def keys(self):
        return self.metadata.keys()


class BalancedSampler(torch.utils.data.sampler.Sampler):
    """Estimate sampling weights in order to rebalance the
    class distributions from an imbalanced dataset.

    Args:
        dataset (BidsDataset): Dataset containing input, gt and metadata.

    Attributes:
        indices (list): List from 0 to length of dataset (number of elements in the dataset).
        nb_samples (int): Number of elements in the dataset.
        weights (Tensor): Weight of each dataset element equal to 1 over the frequency of a given label (inverse of the
                          frequency).
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
        """Returns 1 if sample is not empty, 0 if it is empty (only zeros).

        Args:
            dataset (BidsDataset): Dataset containing input, gt and metadata.
            idx (int): Element index.

        Returns:
            int: 0 or 1.
        """
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
    """Remove keys from metadata. The keys to be deleted are stored in a list.

    Args:
        metadata_lst (list): List of SampleMetadata.

    Returns:
        list: List of SampleMetadata with removed keys.
    """
    metadata_out = []

    if metadata_lst is not None:
        TRANSFORM_PARAMS.remove('crop_params')
        for metadata_cur in metadata_lst:
            for key_ in list(metadata_cur.keys()):
                if key_ in TRANSFORM_PARAMS:
                    del metadata_cur.metadata[key_]
            metadata_out.append(metadata_cur)
        TRANSFORM_PARAMS.append('crop_params')
    return metadata_out


def update_metadata(metadata_src_lst, metadata_dest_lst):
    """Update metadata keys with a reference metadata. A given list of metadata keys will be changed and given the
    values of the reference metadata.

    Args:
        metadata_src_lst (list): List of source metadata used as reference for the destination metadata.
        metadata_dest_lst (list): List of metadate that needs to be updated.

    Returns:
        list: updated metadata list.
    """
    if metadata_src_lst and metadata_dest_lst:
        metadata_dest_lst[0]._update(metadata_src_lst[0], TRANSFORM_PARAMS)
    return metadata_dest_lst

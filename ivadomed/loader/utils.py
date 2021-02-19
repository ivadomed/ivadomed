import collections.abc
import re
import os
import logging
import numpy as np
import pandas as pd
import torch
import joblib
from bids_neuropoly import bids
from sklearn.model_selection import train_test_split
from torch._six import string_classes, int_classes
from ivadomed import utils as imed_utils
import nibabel as nib
import bids as pybids  # "bids" is already taken by bids_neuropoly
import itertools
import random

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

TRANSFORM_PARAMS = ['elastic', 'rotation', 'scale', 'offset', 'crop_params', 'reverse',
                    'translation', 'gaussian_noise']

logger = logging.getLogger(__name__)


def split_dataset(df, center_test_lst, split_method, random_seed, train_frac=0.8, test_frac=0.1):
    """Splits list of subject into training, validation and testing datasets either according to their center or per
    patient. In the 'per_center' option the centers associated the subjects are split according the train, test and
    validation fraction whereas in the 'per_patient', the patients are directly separated according to these fractions.

    Args:
        df (pd.DataFrame): Dataframe containing "participants.tsv" file information.
        center_test_lst (list): list of centers to include in the testing set. Must be
            float.
        split_method (str): Between 'per_center' or 'per_person'. If 'per_center' the separation fraction are
            applied to centers, if 'per_person' they are applied to the subject list.
        random_seed (int): Random seed to ensure reproducible splits.
        train_frac (float): Between 0 and 1. Represents the train set proportion.
        test_frac (float): Between 0 and 1. Represents the test set proportion.
    Returns:
        list, list, list: Train, validation and test subjects list.
    """
    # Init output lists
    X_train, X_val, X_test = [], [], []

    # Split_method cases
    if split_method == 'per_center':
        # make sure that subjects coming from some centers are unseen during training
        if len(center_test_lst) == 0:
            centers = sorted(df['institution_id'].unique().tolist())
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
        # In case we want to use the entire dataset for testing purposes
        X_remain = df['participant_id'].tolist()
        if len(center_test_lst):
            X_test = df[df['institution_id'].isin(center_test_lst)]['participant_id'].tolist()
            X_remain = df[~df['institution_id'].isin(center_test_lst)]['participant_id'].tolist()

        if test_frac == 1 and not len(center_test_lst):
            X_test = df['participant_id'].tolist()
        else:
            X_train, X_remain = train_test_split(X_remain, train_size=train_frac, random_state=random_seed)
            # In case the entire dataset is used to train / validate the model
            if test_frac == 0 or len(center_test_lst):
                X_val = X_remain
            else:
                X_test, X_val = train_test_split(X_remain, train_size=test_frac / (1 - train_frac),
                                                 random_state=random_seed)

    else:
        print(" {split_method} is not a supported split method")

    return X_train, X_val, X_test


def split_dataset_new(df, split_method, data_testing, random_seed, train_frac=0.8, test_frac=0.1):
    """Splits dataset into training, validation and testing sets by applying train, test and validation fractions
    according to the split_method.
    The "data_testing" parameter can be used to specify the data_type and data_value to include in the testing set,
    the dataset is then split as not to mix the data_testing between the training/validation set and the testing set.

    Args:
        df (pd.DataFrame): Dataframe containing all BIDS image files indexed and their metadata.
        split_method (str): Used to specify on which metadata to split the dataset (eg. "participant_id", "sample_id", etc.)
        data_testing (dict): Used to specify data_type and data_value to include in the testing set.
        random_seed (int): Random seed to ensure reproducible splits.
        train_frac (float): Between 0 and 1. Represents the train set proportion.
        test_frac (float): Between 0 and 1. Represents the test set proportion.
    Returns:
        list, list, list: Train, validation and test data_type list.
    """

    # Get data_type and data_value from split parameters
    # If no data_type is provided, data_type is the same as split_method
    data_type = data_testing['data_type'] if data_testing['data_type'] else split_method
    data_value = data_testing['data_value']

    if not split_method in df:
        raise KeyError("No split_method '{}' was not found in metadata".format(split_method))
    if not data_type in df:
        logger.warning("No data_type named '{}' was found in metadata. Not taken into account "
                       "to split the dataset.".format(data_type))
        data_type = split_method

    # Filter dataframe with rows where split_method is not NAN
    df = df[df[split_method].notna()]

    # If no data_value list is provided, create a random data_value according to data_type and test_fraction
    # Split the TEST and remainder set using sklearn function
    if len(data_value) == 0 and test_frac != 0:
        data_value = sorted(df[data_type].unique().tolist())
        test_frac = test_frac if test_frac >= 1 / len(data_value) else 1 / len(data_value)
        data_value, _ = train_test_split(data_value, train_size=test_frac, random_state=random_seed)
    X_test = df[df[data_type].isin(data_value)]['ivadomed_id'].unique().tolist()
    X_remain = df[~df[data_type].isin(data_value)][split_method].unique().tolist()

    # List dataset unique values according to split_method
    # Update train fraction to apply to X_remain
    data = sorted(df[split_method].unique().tolist())
    train_frac_update = train_frac * len(data) / len(X_remain)
    if ((train_frac_update > (1 - 1 / len(X_remain)) and len(X_remain) < 2) or train_frac_update > 1):
        raise RuntimeError("{}/{} '{}' remaining for training and validation sets, train_fraction {} is too large, "
                           "validation set would be empty.".format(len(X_remain), len(data), split_method, train_frac))

    # Split remainder in TRAIN and VALID sets according to train_frac_update using sklearn function
    X_train, X_val = train_test_split(X_remain, train_size=train_frac_update, random_state=random_seed)

    # Convert train and valid sets from list of "split_method" to list of "ivadomed_id"
    X_train = df[df[split_method].isin(X_train)]['ivadomed_id'].unique().tolist()
    X_val = df[df[split_method].isin(X_val)]['ivadomed_id'].unique().tolist()

    # Make sure that test dataset is unseen during training
    # (in cases where there are multiple "data_type" for a same "split_method")
    X_train = list(set(X_train) - set(X_test))
    X_val = list(set(X_val) - set(X_test))

    return X_train, X_val, X_test


def get_new_subject_split(path_data, center_test, split_method, random_seed,
                          train_frac, test_frac, path_output, balance, subject_selection=None):
    """Randomly split dataset between training / validation / testing.

    Randomly split dataset between training / validation / testing\
        and save it in path_output + "/split_datasets.joblib".

    Args:
        path_data (list) or (str): Dataset folders.
        center_test (list): List of centers to include in the testing set.
        split_method (string): See imed_loader_utils.split_dataset.
        random_seed (int): Random seed.
        train_frac (float): Training dataset proportion, between 0 and 1.
        test_frac (float): Testing dataset proportionm between 0 and 1.
        path_output (string): Output folder.
        balance (string): Metadata contained in "participants.tsv" file with categorical values.
            Each category will be evenly distributed in the training, validation and testing
            datasets.
        subject_selection (dict): Used to specify a custom subject selection from a dataset.

    Returns:
        list, list list: Training, validation and testing subjects lists.
    """

    df = merge_bids_datasets(path_data)

    # Save a new merged .tsv on the output folder to be used during evaluation
    df.to_csv(os.path.join(path_output, 'participants.tsv'), sep='\t', index=False)

    if subject_selection is not None:
        # Verify subject_selection format
        if not (len(subject_selection["metadata"]) == len(subject_selection["n"]) == len(subject_selection["value"])):
            raise ValueError("All lists in subject_selection parameter should have the same length.")

        sampled_dfs = []
        for m, n, v in zip(subject_selection["metadata"], subject_selection["n"], subject_selection["value"]):
            sampled_dfs.append(df[df[m] == v].sample(n=n, random_state=random_seed))

        if len(sampled_dfs) != 0:
            df = pd.concat(sampled_dfs)

    # If balance, then split the dataframe for each categorical value of the "balance" column
    if balance:
        if balance in df.keys():
            df_list = [df[df[balance] == k] for k in df[balance].unique().tolist()]
        else:
            logger.warning(f"""No column named '{balance}' was found in 'participants.tsv' file.
                               Not taken into account to split the dataset.""")
            df_list = [df]
    else:
        df_list = [df]

    train_lst, valid_lst, test_lst = [], [], []
    for df_tmp in df_list:
        # Split dataset on each section of subjects
        train_tmp, valid_tmp, test_tmp = split_dataset(df=df_tmp,
                                                       center_test_lst=center_test,
                                                       split_method=split_method,
                                                       random_seed=random_seed,
                                                       train_frac=train_frac,
                                                       test_frac=test_frac)
        # Update the dataset lists
        train_lst += train_tmp
        valid_lst += valid_tmp
        test_lst += test_tmp

    # save the subject distribution
    split_dct = {'train': train_lst, 'valid': valid_lst, 'test': test_lst}
    split_path = os.path.join(path_output, "split_datasets.joblib")
    joblib.dump(split_dct, split_path)

    return train_lst, valid_lst, test_lst


def get_new_subject_split_new(df, split_method, data_testing, random_seed,
                              train_frac, test_frac, path_output, balance, subject_selection=None):
    """Randomly split dataset between training / validation / testing.

    Randomly split dataset between training / validation / testing\
        and save it in path_output + "/split_datasets.joblib".

    Args:
        df (pd.DataFrame): Dataframe containing all BIDS image files indexed and their metadata.
        split_method (str): Used to specify on which metadata to split the dataset (eg. "participant_id", "sample_id", etc.)
        data_testing (dict): Used to specify the data_type and data_value to include in the testing set.
        random_seed (int): Random seed.
        train_frac (float): Training dataset proportion, between 0 and 1.
        test_frac (float): Testing dataset proportionm between 0 and 1.
        path_output (str): Output folder.
        balance (str): Metadata contained in "participants.tsv" file with categorical values. Each category will be
        evenly distributed in the training, validation and testing datasets.
        subject_selection (dict): Used to specify a custom subject selection from a dataset.

    Returns:
        list, list list: Training, validation and testing subjects lists.
    """
    if subject_selection is not None:
        # Verify subject_selection format
        if not (len(subject_selection["metadata"]) == len(subject_selection["n"]) == len(subject_selection["value"])):
            raise ValueError("All lists in subject_selection parameter should have the same length.")

        sampled_dfs = []
        random.seed(random_seed)
        for m, n, v in zip(subject_selection["metadata"], subject_selection["n"], subject_selection["value"]):
            participants = random.sample(df[df[m] == v]['participant_id'].unique().tolist(), n)
            for participant in participants:
                sampled_dfs.append(df[df['participant_id'] == participant])

        if len(sampled_dfs) != 0:
            df = pd.concat(sampled_dfs)

    # If balance, then split the dataframe for each categorical value of the "balance" column
    if balance:
        if balance in df.keys():
            df_list = [df[df[balance] == k] for k in df[balance][df[balance].notna()].unique().tolist()]
        else:
            logger.warning("No column named '{}' was found in 'participants.tsv' file. Not taken into account to split "
                           "the dataset.".format(balance))
            df_list = [df]
    else:
        df_list = [df]

    train_lst, valid_lst, test_lst = [], [], []
    for df_tmp in df_list:
        # Split dataset on each section of subjects
        train_tmp, valid_tmp, test_tmp = split_dataset_new(df=df_tmp,
                                                           split_method=split_method,
                                                           data_testing=data_testing,
                                                           random_seed=random_seed,
                                                           train_frac=train_frac,
                                                           test_frac=test_frac)
        # Update the dataset lists
        train_lst += train_tmp
        valid_lst += valid_tmp
        test_lst += test_tmp

    # save the subject distribution
    split_dct = {'train': train_lst, 'valid': valid_lst, 'test': test_lst}
    split_path = os.path.join(path_output, "split_datasets.joblib")
    joblib.dump(split_dct, split_path)

    return train_lst, valid_lst, test_lst


def get_subdatasets_subjects_list(split_params, path_data, path_output, subject_selection=None):
    """Get lists of subjects for each sub-dataset between training / validation / testing.

    Args:
        split_params (dict): Split parameters, see :doc:`configuration_file` for more details.
        path_data (list): Path to the BIDS dataset(s).
        path_output (str): Output folder.
        subject_selection (dict): Used to specify a custom subject selection from a dataset.

    Returns:
        list, list list: Training, validation and testing subjects lists.
    """
    if split_params["fname_split"]:
        # Load subjects lists
        old_split = joblib.load(split_params["fname_split"])
        train_lst, valid_lst, test_lst = old_split['train'], old_split['valid'], old_split['test']
    else:
        train_lst, valid_lst, test_lst = get_new_subject_split(path_data=path_data,
                                                               center_test=split_params['center_test'],
                                                               split_method=split_params['method'],
                                                               random_seed=split_params['random_seed'],
                                                               train_frac=split_params['train_fraction'],
                                                               test_frac=split_params['test_fraction'],
                                                               path_output=path_output,
                                                               balance=split_params['balance']
                                                               if 'balance' in split_params else None,
                                                               subject_selection=subject_selection)
    return train_lst, valid_lst, test_lst


def get_subdatasets_subjects_list_new(split_params, df, path_output, subject_selection=None):
    """Get lists of subjects for each sub-dataset between training / validation / testing.

    Args:
        split_params (dict): Split parameters, see :doc:`configuration_file` for more details.
        df (pd.DataFrame): Dataframe containing all BIDS image files indexed and their metadata.
        path_output (str): Output folder.
        subject_selection (dict): Used to specify a custom subject selection from a dataset.

    Returns:
        list, list list: Training, validation and testing subjects lists.
    """
    if split_params["fname_split"]:
        # Load subjects lists
        old_split = joblib.load(split_params["fname_split"])
        train_lst, valid_lst, test_lst = old_split['train'], old_split['valid'], old_split['test']
    else:
        train_lst, valid_lst, test_lst = get_new_subject_split_new(df=df,
                                                                   split_method=split_params['split_method'],
                                                                   data_testing=split_params['data_testing'],
                                                                   random_seed=split_params['random_seed'],
                                                                   train_frac=split_params['train_fraction'],
                                                                   test_frac=split_params['test_fraction'],
                                                                   path_output=path_output,
                                                                   balance=split_params['balance']
                                                                   if 'balance' in split_params else None,
                                                                   subject_selection=subject_selection)
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
    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: imed_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.abc.Sequence):
        return [imed_collate(samples) for samples in batch]

    return batch


def filter_roi(roi_data, nb_nonzero_thr):
    """Filter slices from dataset using ROI data.

    This function filters slices (roi_data) where the number of non-zero voxels within the
    ROI slice (e.g. centerline, SC segmentation) is inferior or equal to a given threshold
    (nb_nonzero_thr).

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
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.

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
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.

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
        data (list or tuple): Shape or numbers associated with each image dimension
            (e.g. image resolution).
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.

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
        """Update metadata keys with a reference metadata.

        A given list of metadata keys will be changed and given the values of the reference
        metadata.

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
        metadata (str): Indicates which metadata to use to balance the sampler.

    Attributes:
        indices (list): List from 0 to length of dataset (number of elements in the dataset).
        nb_samples (int): Number of elements in the dataset.
        weights (Tensor): Weight of each dataset element equal to 1 over the frequency of a
            given label (inverse of the frequency).
        metadata_dict (dict): Stores the mapping from metadata string to index (int).
        label_idx (int): Keeps track of the label indices already used for the metadata_dict.
    """

    def __init__(self, dataset, metadata='gt'):
        self.indices = list(range(len(dataset)))

        self.nb_samples = len(self.indices)
        self.metadata_dict = {}
        self.label_idx = 0

        cmpt_label = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx, metadata)
            if label in cmpt_label:
                cmpt_label[label] += 1
            else:
                cmpt_label[label] = 1

        weights = [1.0 / cmpt_label[self._get_label(dataset, idx, metadata)]
                   for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx, metadata):
        """Returns 1 if sample is not empty, 0 if it is empty (only zeros).

        Args:
            dataset (BidsDataset): Dataset containing input, gt and metadata.
            idx (int): Element index.

        Returns:
            int: 0 or 1.
        """
        if metadata != 'gt':
            label_str = dataset[idx]['input_metadata'][0][metadata]
            if label_str not in self.metadata_dict:
                self.metadata_dict[label_str] = self.label_idx
                self.label_idx += 1
            return self.metadata_dict[label_str]

        else:
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
    """Update metadata keys with a reference metadata.

    A given list of metadata keys will be changed and given the values of the reference metadata.

    Args:
        metadata_src_lst (list): List of source metadata used as reference for the
            destination metadata.
        metadata_dest_lst (list): List of metadate that needs to be updated.

    Returns:
        list: updated metadata list.
    """
    if metadata_src_lst and metadata_dest_lst:
        if not isinstance(metadata_dest_lst[0], list):  # annotation from one rater only
            metadata_dest_lst[0]._update(metadata_src_lst[0], TRANSFORM_PARAMS)
        else:  # annotations from several raters
            for idx, _ in enumerate(metadata_dest_lst[0]):
                metadata_dest_lst[0][idx]._update(metadata_src_lst[0], TRANSFORM_PARAMS)
    return metadata_dest_lst


class SliceFilter(object):
    """Filter 2D slices from dataset.

    If a sample does not meet certain conditions, it is discarded from the dataset.

    Args:
        filter_empty_mask (bool): If True, samples where all voxel labels are zeros are discarded.
        filter_empty_input (bool): If True, samples where all voxel intensities are zeros are discarded.
        filter_absent_class (bool): If True, samples where all voxel labels are zero for one or more classes are discarded.
        filter_classification (bool): If True, samples where all images fail a custom classifier filter are discarded.

    Attributes:
        filter_empty_mask (bool): If True, samples where all voxel labels are zeros are discarded.
        filter_empty_input (bool): If True, samples where all voxel intensities are zeros are discarded.
        filter_absent_class (bool): If True, samples where all voxel labels are zero for one or more classes are discarded.
        filter_classification (bool): If True, samples where all images fail a custom classifier filter are discarded.

    """

    def __init__(self, filter_empty_mask=True,
                 filter_empty_input=True,
                 filter_classification=False,
                 filter_absent_class=False,
                 classifier_path=None, device=None, cuda_available=None):
        self.filter_empty_mask = filter_empty_mask
        self.filter_empty_input = filter_empty_input
        self.filter_absent_class = filter_absent_class
        self.filter_classification = filter_classification
        self.device = device
        self.cuda_available = cuda_available

        if self.filter_classification:
            if cuda_available:
                self.classifier = torch.load(classifier_path, map_location=device)
            else:
                self.classifier = torch.load(classifier_path, map_location='cpu')

    def __call__(self, sample):
        input_data, gt_data = sample['input'], sample['gt']

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


def reorient_image(arr, slice_axis, nib_ref, nib_ref_canonical):
    """Reorient an image to match a reference image orientation.

    It reorients a array to a given orientation and convert it to a nibabel object using the
    reference nibabel header.

    Args:
        arr (ndarray): Input array, array to re orient.
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.
        nib_ref (nibabel): Reference nibabel object, whose header is used.
        nib_ref_canonical (nibabel): `nib_ref` that has been reoriented to canonical orientation (RAS).
    """
    # Orient image in RAS according to slice axis
    arr_ras = orient_img_ras(arr, slice_axis)

    # https://gitship.com/neuroscience/nibabel/blob/master/nibabel/orientations.py
    ref_orientation = nib.orientations.io_orientation(nib_ref.affine)
    ras_orientation = nib.orientations.io_orientation(nib_ref_canonical.affine)
    # Return the orientation that transforms from ras to ref_orientation
    trans_orient = nib.orientations.ornt_transform(ras_orientation, ref_orientation)
    # apply transformation
    return nib.orientations.apply_orientation(arr_ras, trans_orient)


def merge_bids_datasets(path_data):
    """Read the participants.tsv from several BIDS folders and merge them into a single dataframe.
    Args:
        path_data (list) or (str): BIDS folders paths

    Returns:
        df: dataframe with merged subjects and columns
    """
    path_data = imed_utils.format_path_data(path_data)

    if len(path_data) == 1:
        # read participants.tsv as pandas dataframe
        df = bids.BIDS(path_data[0]).participants.content
        # Append a new column to show which dataset the Subjects belong to (this will be used later for loading)
        df['path_output'] = [path_data[0]] * len(df)
    elif path_data == []:
        raise Exception("No dataset folder selected")
    else:
        # Merge multiple .tsv files into the same dataframe
        df = pd.read_table(os.path.join(path_data[0], 'participants.tsv'), encoding="ISO-8859-1")
        # Convert to string to get rid of potential TypeError during merging within the same column
        df = df.astype(str)

        # Add the Bids_path to the dataframe
        df['path_output'] = [path_data[0]] * len(df)

        for iFolder in range(1, len(path_data)):
            df_next = pd.read_table(os.path.join(path_data[iFolder], 'participants.tsv'),
                                    encoding="ISO-8859-1")
            df_next = df_next.astype(str)
            df_next['path_output'] = [path_data[iFolder]] * len(df_next)
            # Merge the .tsv files (This keeps also non-overlapping fields)
            df = pd.merge(left=df, right=df_next, how='outer')

    # Get rid of duplicate entries based on the field "participant_id" (the same subject could have in theory be
    # included in both datasets). The assumption here is that if the two datasets contain the same subject,
    # identical sessions of the subjects are contained within the two folder so only the files within the first folder
    # will be kept.
    logical_keep_first_encounter = []
    indicesOfDuplicates = []
    used = set()  # For debugging

    for iEntry in range(len(df)):
        if df['participant_id'][iEntry] not in used:
            used.add(df['participant_id'][iEntry])  # For debugging
            logical_keep_first_encounter.append(iEntry)
        else:
            indicesOfDuplicates.append(iEntry)  # For debugging
    # Just keep the dataframe with unique participant_id
    df = df.iloc[logical_keep_first_encounter, :]

    # Rearrange the bids paths to be last column of the dataframe
    cols = list(df.columns.values)
    cols.remove("path_output")
    cols.append("path_output")
    df = df[cols]

    # Substitute NaNs with string: "-". This helps with metadata selection
    df = df.fillna("-")

    return df


class BidsDataframe:
    """
    This class aims to create a dataframe containing all BIDS image files in a path_data and their metadata.

    Args:
        loader_params (dict): Loader parameters, see :doc:`configuration_file` for more details.
        derivatives (bool): If True, derivatives are indexed.
        path_output (str): Output folder.

    Attributes:
        path_data (str): Path to the BIDS dataset.
        bids_config (str): Path to the custom BIDS configuration file.
        target_suffix (list of str): List of suffix of targetted structures.
        roi_suffix (str): List of suffix of ROI masks.
        extensions (list of str): List of file extensions of interest.
        contrast_lst (list of str): List of the contrasts of interest.
        derivatives (bool): If True, derivatives are indexed.
        df (pd.DataFrame): Dataframe containing dataset information
    """

    def __init__(self, loader_params, derivatives, path_output):

        # path_data from loader parameters
        self.paths_data = imed_utils.format_path_data(loader_params['path_data'])

        # bids_config from loader parameters
        self.bids_config = None if 'bids_config' not in loader_params else loader_params['bids_config']

        # target_suffix and roi_suffix from loader parameters
        self.target_suffix = loader_params['target_suffix']
        # If `target_suffix` is a list of lists convert to list
        if any(isinstance(t, list) for t in self.target_suffix):
            self.target_suffix = list(itertools.chain.from_iterable(self.target_suffix))
        self.roi_suffix = loader_params['roi_params']['suffix']
        # If `roi_suffix` is not None, add to target_suffix
        if self.roi_suffix is not None:
            self.target_suffix.append(self.roi_suffix)

        # extensions from loader parameters
        self.extensions = loader_params['extensions']

        # contrast_lst from loader parameters
        self.contrast_lst = loader_params["contrast_params"]["contrast_lst"]

        # derivatives
        self.derivatives = derivatives

        # Create dataframe
        self.df = pd.DataFrame()
        self.create_bids_dataframe()

        # Save dataframe as csv file
        self.save(os.path.join(path_output, "bids_dataframe.csv"))

    def create_bids_dataframe(self):
        """Generate the dataframe."""

        # Suppress a Future Warning from pybids about leading dot included in 'extension' from version 0.14.0
        # The config_bids.json file used matches the future behavior
        # TODO: when reaching version 0.14.0, remove the following line
        pybids.config.set_option('extension_initial_dot', True)

        for path_data in self.paths_data:
            path_data = os.path.join(path_data, '')

            # Initialize BIDSLayoutIndexer and BIDSLayout
            # validate=True by default for both indexer and layout, BIDS-validator is not skipped
            # Force index for samples tsv and json files, and for subject subfolders containing microscopy files based on extensions.
            # TODO: remove force indexing of microscopy files after BEP microscopy is merged in BIDS
            ext_microscopy = ('.png', '.ome.tif', '.ome.tiff', '.ome.tf2', '.ome.tf8', '.ome.btf')
            force_index = ['samples.tsv', 'samples.json']
            for root, dirs, files in os.walk(path_data):
                for file in files:
                    if file.endswith(ext_microscopy) and (root.replace(path_data, '').startswith("sub")):
                        force_index.append(os.path.join(root.replace(path_data, '')))
            indexer = pybids.BIDSLayoutIndexer(force_index=force_index)
            layout = pybids.BIDSLayout(path_data, config=self.bids_config, indexer=indexer,
                                       derivatives=self.derivatives)

            # Transform layout to dataframe with all entities and json metadata
            # As per pybids, derivatives don't include parsed entities, only the "path" column
            df_next = layout.to_df(metadata=True)

            # Add filename and parent_path columns
            df_next['filename'] = df_next['path'].apply(os.path.basename)
            df_next['parent_path'] = df_next['path'].apply(os.path.dirname)

            # Drop rows with json, tsv and LICENSE files in case no extensions are provided in config file for filtering
            df_next = df_next[~df_next['filename'].str.endswith(tuple(['.json', '.tsv', 'LICENSE']))]

            # Add ivadomed_id column corresponding to filename minus modality and extension for files that are not derivatives.
            for index, row in df_next.iterrows():
                if isinstance(row['suffix'], str):
                    df_next.loc[index, 'ivadomed_id'] = re.sub(r'_' + row['suffix'] + '.*', '', row['filename'])

            # Update dataframe with subject files of chosen contrasts and extensions,
            # and with derivative files of chosen target_suffix from loader parameters
            df_next = df_next[(~df_next['path'].str.contains('derivatives')
                               & df_next['suffix'].str.contains('|'.join(self.contrast_lst))
                               & df_next['extension'].str.contains('|'.join(self.extensions)))
                               | (df_next['path'].str.contains('derivatives')
                               & df_next['filename'].str.contains('|'.join(self.target_suffix)))]

            # Add tsv files metadata to dataframe
            df_next = self.add_tsv_metadata(df_next, path_data, layout)

            # TODO: check if other files are needed for EEG and DWI

            # Merge dataframes
            self.df = pd.concat([self.df, df_next], join='outer', ignore_index=True)

        # Drop duplicated rows based on all columns except 'path and 'parent_path'
        # Keep first occurence
        columns = self.df.columns.to_list()
        columns.remove('path')
        columns.remove('parent_path')
        self.df = self.df[~(self.df.astype(str).duplicated(subset=columns, keep='first'))]

        # If indexing of derivatives is true
        if self.derivatives:

            # Get list of subject files with available derivatives
            has_deriv, deriv = self.get_subjects_with_derivatives()

            # Filter dataframe to keep subjects files with available derivatives only
            if has_deriv:
                self.df = self.df[self.df['filename'].str.contains('|'.join(has_deriv))
                                  | self.df['filename'].str.contains('|'.join(deriv))]
            else:
                # Raise error and exit if no derivatives are found for any subject files
                raise RuntimeError("Derivatives not found.")

        # Reset index
        self.df.reset_index(drop=True, inplace=True)

        # Drop columns with all null values
        self.df.dropna(axis=1, inplace=True, how='all')

    def add_tsv_metadata(self, df, path_data, layout):

        """Add tsv files metadata to dataframe.
        Args:
            layout (BIDSLayout): pybids BIDSLayout of the indexed files of the path_data
        """

        # Add participant_id column, and metadata from participants.tsv file if present
        # Uses pybids function
        df['participant_id'] = "sub-" + df['subject']
        if layout.get_collections(level='dataset'):
            df_participants = layout.get_collections(level='dataset', merge=True).to_df()
            df_participants.drop(['suffix'], axis=1, inplace=True)
            df = pd.merge(df, df_participants, on='subject', suffixes=("_x", None), how='left')

        # Add sample_id column if sample column exists, and add metadata from samples.tsv file if present
        # TODO: use pybids function after BEP microscopy is merged in BIDS
        if 'sample' in df:
            df['sample_id'] = "sample-" + df['sample']
        fname_samples = os.path.join(path_data, "samples.tsv")
        if os.path.exists(fname_samples):
            df_samples = pd.read_csv(fname_samples, sep='\t')
            df = pd.merge(df, df_samples, on=['participant_id', 'sample_id'], suffixes=("_x", None),
                               how='left')

        # Add metadata from all _sessions.tsv files, if present
        # Uses pybids function
        if layout.get_collections(level='subject'):
            df_sessions = layout.get_collections(level='subject', merge=True).to_df()
            df_sessions.drop(['suffix'], axis=1, inplace=True)
            df = pd.merge(df, df_sessions, on=['subject', 'session'], suffixes=("_x", None), how='left')

        # Add metadata from all _scans.tsv files, if present
        # TODO: use pybids function after BEP microscopy is merged in BIDS
        # TODO: verify merge behavior with EEG and DWI scans files, tested with anat and microscopy only
        df_scans = pd.DataFrame()
        for root, dirs, files in os.walk(path_data):
            for file in files:
                if file.endswith("scans.tsv"):
                    df_temp = pd.read_csv(os.path.join(root, file), sep='\t')
                    df_scans = pd.concat([df_scans, df_temp], ignore_index=True)
        if not df_scans.empty:
            df_scans['filename'] = df_scans['filename'].apply(os.path.basename)
            df = pd.merge(df, df_scans, on=['filename'], suffixes=("_x", None), how='left')

        return df

    def get_subjects_with_derivatives(self):
        """Get lists of subject filenames with available derivatives.

        Returns:
            list, list: subject filenames having derivatives, available derivatives filenames.
        """
        subject_fnames = self.get_subject_fnames()
        deriv_fnames = self.get_deriv_fnames()
        has_deriv = []
        deriv = []

        for subject_fname in subject_fnames:
            available = self.get_derivatives(subject_fname, deriv_fnames)
            if available:
                if self.roi_suffix is not None:
                    if self.roi_suffix in ('|'.join(available)):
                        has_deriv.append(subject_fname)
                        deriv.extend(available)
                    else:
                        logger.warning("Missing roi_suffix {} for {}. Skipping."
                                       .format(self.roi_suffix, subject_fname))
                else:
                    has_deriv.append(subject_fname)
                    deriv.extend(available)
                for target in self.target_suffix:
                    if target not in str(available) and target != self.roi_suffix:
                        logger.warning("Missing target_suffix {} for {}".format(target, subject_fname))
            else:
                logger.warning("Missing derivatives for {}. Skipping.".format(subject_fname))

        return has_deriv, deriv

    def get_subject_fnames(self):
        """Get the list of subject filenames in dataframe.

        Returns:
            list: subject filenames.
        """
        return self.df[~self.df['path'].str.contains('derivatives')]['filename'].to_list()

    def get_deriv_fnames(self):
        """Get the list of derivative filenames in dataframe.

        Returns:
            list: derivative filenames.
        """
        return self.df[self.df['path'].str.contains('derivatives')]['filename'].tolist()

    def get_derivatives(self, subject_fname, deriv_fnames):
        """Return list of available derivative filenames for a subject filename.
        Args:
            subject_fname (str): Subject filename.
            deriv_fnames (list of str): List of derivative filenames.

        Returns:
            list: derivative filenames
        """
        prefix_fname = subject_fname.split('.')[0]
        return [d for d in deriv_fnames if prefix_fname in d]

    def save(self, path):
        """Save the dataframe into a csv file.
        Args:
            path (str): Path to csv file.
        """
        try:
            self.df.to_csv(path, index=False)
            print("Dataframe has been saved at {}.".format(path))
        except FileNotFoundError:
            print("Wrong path.")

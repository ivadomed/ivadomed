import collections
import re
import os
import logging

import numpy as np
import torch
import joblib
from bids_neuropoly import bids
from sklearn.model_selection import train_test_split
from torch._six import string_classes, int_classes
from ivadomed import utils as imed_utils
import nibabel as nib

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

logger = logging.getLogger(__name__)


def split_dataset(df, center_test_lst, split_method, random_seed, train_frac=0.8, test_frac=0.1):
    """Splits list of subject into training, validation and testing datasets either according to their center or per
    patient. In the 'per_center' option the centers associated the subjects are split according the train, test and
    validation fraction whereas in the 'per_patient', the patients are directly separated according to these fractions.

    Args:
        df (pd.DataFrame): Dataframe containing "participants.tsv" file information.
        center_test_lst (list): list of centers to include in the testing set.
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
        if test_frac == 1:
            X_test = df['participant_id'].tolist()
        else:
            X_train, X_remain = train_test_split(df['participant_id'].tolist(), train_size=train_frac,
                                                 random_state=random_seed)
            # In case the entire dataset is used to train / validate the model
            if test_frac == 0:
                X_val = X_remain
            else:
                X_test, X_val = train_test_split(X_remain, train_size=test_frac / (1 - train_frac),
                                                 random_state=random_seed)

    else:
        print(" {split_method} is not a supported split method")

    return X_train, X_val, X_test


def get_new_subject_split(path_folder, center_test, split_method, random_seed,
                          train_frac, test_frac, log_directory, balance):
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
        balance (string): Metadata contained in "participants.tsv" file with categorical values. Each category will be
        evenly distributed in the training, validation and testing datasets.

    Returns:
        list, list list: Training, validation and testing subjects lists.
    """
    # read participants.tsv as pandas dataframe
    df = bids.BIDS(path_folder).participants.content

    # If balance, then split the dataframe for each categorical value of the "balance" column
    if balance:
        if balance in df.keys():
            df_list = [df[df[balance] == k] for k in df[balance].unique().tolist()]
        else:
            logger.warning("No column named '{}' was found in 'participants.tsv' file. Not taken into account to split "
                           "the dataset.".format(balance))
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
    split_path = os.path.join(log_directory, "split_datasets.joblib")
    joblib.dump(split_dct, split_path)

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
                                                               log_directory=log_directory,
                                                               balance=split_params['balance']
                                                               if 'balance' in split_params else None)
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


class SliceFilter(object):
    """Filter 2D slices from dataset.

    If a sample does not meet certain conditions, it is discarded from the dataset.

    Args:
        filter_empty_mask (bool): If True, samples where all voxel labels are zeros are discarded.
        filter_empty_input (bool): If True, samples where all voxel intensities are zeros are discarded.

    Attributes:
        filter_empty_mask (bool): If True, samples where all voxel labels are zeros are discarded.
        filter_empty_input (bool): If True, samples where all voxel intensities are zeros are discarded.
    """

    def __init__(self, filter_empty_mask=True,
                 filter_empty_input=True,
                 filter_classification=False, classifier_path=None, device=None, cuda_available=None):
        self.filter_empty_mask = filter_empty_mask
        self.filter_empty_input = filter_empty_input
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
            if not np.any(gt_data):
                return False

        if self.filter_empty_input:
            # Filter set of images if one of them is empty or filled with constant value (i.e. std == 0)
            if np.any([img.std() == 0 for img in input_data]):
                return False

        if self.filter_classification:
            if not np.all([int(
                    self.classifier(imed_utils.cuda(torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(0), self.cuda_available)))
                           for img in input_data]):
                return False

        return True


def reorient_image(arr, slice_axis, nib_ref, nib_ref_canonical):
    """Reorient an image to match a reference image orientation.

    It reorients a array to a given orientation and convert it to a nibabel object using the reference nibabel header.

    Args:
        arr (ndarray): Input array, array to re orient.
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
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

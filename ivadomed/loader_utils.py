import collections
import json
import os
import re
from copy import deepcopy

import numpy as np
import torch
from bids_neuropoly import bids
from scipy.signal import argrelextrema
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import OneHotEncoder
from torch._six import string_classes, int_classes

from ivadomed import adaptative as imed_adaptative
from ivadomed import loader as imed_loader
from ivadomed import utils as imed_utils
from ivadomed import __path__

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

with open(os.path.join(__path__[0], "config/contrast_dct.json"), "r") as fhandle:
    GENERIC_CONTRAST = json.load(fhandle)
MANUFACTURER_CATEGORY = {'Siemens': 0, 'Philips': 1, 'GE': 2}
CONTRAST_CATEGORY = {"T1w": 0, "T2w": 1, "T2star": 2,
                     "acq-MToff_MTS": 3, "acq-MTon_MTS": 4, "acq-T1w_MTS": 5}


def load_dataset(data_list, data_transform, context):
    if context["unet_3D"]:
        dataset = imed_loader.Bids3DDataset(context["bids_path"],
                                            subject_lst=data_list,
                                            target_suffix=context["target_suffix"],
                                            roi_suffix=context["roi_suffix"],
                                            contrast_lst=context["contrast_train_validation"],
                                            metadata_choice=context["metadata"],
                                            contrast_balance=context["contrast_balance"],
                                            slice_axis=imed_utils.AXIS_DCT[context["slice_axis"]],
                                            transform=data_transform,
                                            multichannel=context['multichannel'],
                                            length=context["length_3D"],
                                            padding=context["padding_3D"])
    elif context["HeMIS"]:
        dataset = imed_adaptative.HDF5Dataset(root_dir=context["bids_path"],
                                              subject_lst=data_list,
                                              hdf5_name=context["hdf5_path"],
                                              csv_name=context["csv_path"],
                                              target_suffix=context["target_suffix"],
                                              contrast_lst=context["contrast_train_validation"],
                                              ram=context['ram'],
                                              contrast_balance=context["contrast_balance"],
                                              slice_axis=imed_utils.AXIS_DCT[context["slice_axis"]],
                                              transform=data_transform,
                                              metadata_choice=context["metadata"],
                                              slice_filter_fn=imed_utils.SliceFilter(**context["slice_filter"]),
                                              roi_suffix=context["roi_suffix"],
                                              target_lst=context['target_lst'],
                                              roi_lst=context['roi_lst'])
    else:
        dataset = imed_loader.BidsDataset(context["bids_path"],
                                          subject_lst=data_list,
                                          target_suffix=context["target_suffix"],
                                          roi_suffix=context["roi_suffix"],
                                          contrast_lst=context["contrast_train_validation"],
                                          metadata_choice=context["metadata"],
                                          contrast_balance=context["contrast_balance"],
                                          slice_axis=imed_utils.AXIS_DCT[context["slice_axis"]],
                                          transform=data_transform,
                                          multichannel=context['multichannel'],
                                          slice_filter_fn=imed_utils.SliceFilter(**context["slice_filter"]))

    return dataset


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
        print(f" {split_method} is not a supported split method")

    return X_train, X_val, X_test


def mt_collate(batch):
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
        return {key: mt_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [mt_collate(samples) for samples in transposed]

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


class SampleMetadata(object):
    def __init__(self, d=None):
        self.metadata = {} or d

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def __getitem__(self, key):
        return self.metadata[key]

    def __contains__(self, key):
        return key in self.metadata

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


def normalize_metadata(ds_in, clustering_models, debugging, metadata_type, train_set=False):
    if train_set:
        # Initialise One Hot Encoder
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_train_ohe = []

    ds_out = []
    for idx, subject in enumerate(ds_in):
        s_out = deepcopy(subject)
        if metadata_type == 'mri_params':
            # categorize flip angle, repetition time and echo time values using KDE
            for m in ['FlipAngle', 'RepetitionTime', 'EchoTime']:
                v = subject["input_metadata"][m]
                p = clustering_models[m].predict(v)
                s_out["input_metadata"][m] = p
                if debugging:
                    print("{}: {} --> {}".format(m, v, p))

            # categorize manufacturer info based on the MANUFACTURER_CATEGORY dictionary
            manufacturer = subject["input_metadata"]["Manufacturer"]
            if manufacturer in MANUFACTURER_CATEGORY:
                s_out["input_metadata"]["Manufacturer"] = MANUFACTURER_CATEGORY[manufacturer]
                if debugging:
                    print("Manufacturer: {} --> {}".format(manufacturer,
                                                           MANUFACTURER_CATEGORY[manufacturer]))
            else:
                print("{} with unknown manufacturer.".format(subject))
                # if unknown manufacturer, then value set to -1
                s_out["input_metadata"]["Manufacturer"] = -1

            s_out["input_metadata"]["film_input"] = [s_out["input_metadata"][k] for k in
                                                     ["FlipAngle", "RepetitionTime", "EchoTime", "Manufacturer"]]
        else:
            for i, input_metadata in enumerate(subject["input_metadata"]):
                generic_contrast = GENERIC_CONTRAST[input_metadata["contrast"]]
                label_contrast = CONTRAST_CATEGORY[generic_contrast]
                s_out["input_metadata"][i]["film_input"] = [label_contrast]

        for i, input_metadata in enumerate(subject["input_metadata"]):
            s_out["input_metadata"][i]["contrast"] = input_metadata["contrast"]

            if train_set:
                X_train_ohe.append(s_out["input_metadata"][i]["film_input"])
            ds_out.append(s_out)

        del s_out, subject

    if train_set:
        X_train_ohe = np.vstack(X_train_ohe)
        ohe.fit(X_train_ohe)
        return ds_out, ohe
    else:
        return ds_out


class Kde_model():
    def __init__(self):
        self.kde = KernelDensity()
        self.minima = None

    def train(self, data, value_range, gridsearch_bandwidth_range):
        # reshape data to fit sklearn
        data = np.array(data).reshape(-1, 1)

        # use grid search cross-validation to optimize the bandwidth
        params = {'bandwidth': gridsearch_bandwidth_range}
        grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
        grid.fit(data)

        # use the best estimator to compute the kernel density estimate
        self.kde = grid.best_estimator_

        # fit kde with the best bandwidth
        self.kde.fit(data)

        s = value_range
        e = self.kde.score_samples(s.reshape(-1, 1))

        # find local minima
        self.minima = s[argrelextrema(e, np.less)[0]]

    def predict(self, data):
        x = [i for i, m in enumerate(self.minima) if data < m]
        pred = min(x) if len(x) else len(self.minima)
        return pred


def clustering_fit(dataset, key_lst):
    """This function creates clustering models for each metadata type,
    using Kernel Density Estimation algorithm.
    :param datasets (list): data
    :param key_lst (list of strings): names of metadata to cluster
    :return: clustering model for each metadata type
    """
    KDE_PARAM = {'FlipAngle': {'range': np.linspace(0, 360, 1000), 'gridsearch': np.logspace(-4, 1, 50)},
                 'RepetitionTime': {'range': np.logspace(-1, 1, 1000), 'gridsearch': np.logspace(-4, 1, 50)},
                 'EchoTime': {'range': np.logspace(-3, 1, 1000), 'gridsearch': np.logspace(-4, 1, 50)}}

    model_dct = {}
    for k in key_lst:
        k_data = [value for value in dataset[k]]

        kde = Kde_model()
        kde.train(k_data, KDE_PARAM[k]['range'], KDE_PARAM[k]['gridsearch'])

        model_dct[k] = kde

    return model_dct

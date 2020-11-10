import json
import os
from copy import deepcopy

import numpy as np
from scipy.signal import argrelextrema
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import OneHotEncoder

from ivadomed import __path__

with open(os.path.join(__path__[0], "config/contrast_dct.json"), "r") as fhandle:
    GENERIC_CONTRAST = json.load(fhandle)
MANUFACTURER_CATEGORY = {'Siemens': 0, 'Philips': 1, 'GE': 2}
CONTRAST_CATEGORY = {"T1w": 0, "T2w": 1, "T2star": 2,
                     "acq-MToff_MTS": 3, "acq-MTon_MTS": 4, "acq-T1w_MTS": 5}


def normalize_metadata(ds_in, clustering_models, debugging, metadata_type, train_set=False):
    """Categorize each metadata value using a KDE clustering method, then apply a one-hot-encoding.

    Args:
         ds_in (BidsDataset): Dataset with metadata.
         clustering_models: Pre-trained clustering model that has been trained on metadata of the training set.
         debugging (bool): If True, extended verbosity and intermediate outputs.
         metadata_type (str): Choice between 'mri_params', 'constrasts' or the name of a column from the
            participants.tsv file.
         train_set (bool): Indicates if the input dataset is the training dataset (True) or the validation or testing
            dataset (False).

    Returns:
        BidsDataset: Dataset with normalized metadata. If train_set is True, then the one-hot-encoder model is also
            returned.
    """
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
        elif metadata_type == "contrasts":
            for i, input_metadata in enumerate(subject["input_metadata"]):
                generic_contrast = GENERIC_CONTRAST[input_metadata["contrast"]]
                label_contrast = CONTRAST_CATEGORY[generic_contrast]
                s_out["input_metadata"][i]["film_input"] = [label_contrast]
        else:
            for i, input_metadata in enumerate(subject["input_metadata"]):
                data = input_metadata[metadata_type]
                label_contrast = input_metadata['metadata_dict'][data]
                s_out["input_metadata"][i]["film_input"] = [label_contrast]

        for i, input_metadata in enumerate(subject["input_metadata"]):
            if 'contrast' in input_metadata:
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
    """Kernel Density Estimation.

    Apply this clustering method to metadata values, using (`sklearn implementation.
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity>`__)

    Attributes:
        kde (sklearn.neighbors.KernelDensity):
        minima (float): Local minima.
    """
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

    Args:
        datasets (list): data
        key_lst (list of str): names of metadata to cluster

    Returns:
        dict: Clustering model for each metadata type in a dictionary where the keys are the metadata names.
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


def check_isMRIparam(mri_param_type, mri_param, subject, metadata):
    """Check if a given metadata belongs to the MRI parameters.

    Args:
        mri_param_type (str): Metadata type name.
        mri_param (list): List of MRI params names.
        subject (str): Current subject name.
        metadata (dict): Metadata.

    Returns:
        bool: True if `mri_param_type` is part of `mri_param`.
    """
    if mri_param_type not in mri_param:
        print("{} without {}, skipping.".format(subject, mri_param_type))
        return False
    else:
        if mri_param_type == "Manufacturer":
            value = mri_param[mri_param_type]
        else:
            if isinstance(mri_param[mri_param_type], (int, float)):
                value = float(mri_param[mri_param_type])
            else:  # eg multi-echo data have 3 echo times
                value = np.mean([float(v)
                                 for v in mri_param[mri_param_type].split(',')])

        metadata[mri_param_type].append(value)
        return True


def get_film_metadata_models(ds_train, metadata_type, debugging=False):
    """Get FiLM models.

    This function pulls the clustering and one-hot encoder models that are used by FiLMedUnet.
    It also calls the normalization of metadata.

    Args:
        ds_train (MRI2DSegmentationDataset): training dataset
        metadata_type (string): eg mri_params, contrasts
        debugging (bool):

    Returns:
        MRI2DSegmentationDataset, OneHotEncoder, KernelDensity: dataset, one-hot encoder and KDE model
    """
    if metadata_type == "mri_params":
        metadata_vector = ["RepetitionTime", "EchoTime", "FlipAngle"]
        metadata_clustering_models = clustering_fit(ds_train.metadata, metadata_vector)
    else:
        metadata_clustering_models = None

    ds_train, train_onehotencoder = normalize_metadata(ds_train,
                                                       metadata_clustering_models,
                                                       debugging,
                                                       metadata_type,
                                                       True)

    return ds_train, train_onehotencoder, metadata_clustering_models

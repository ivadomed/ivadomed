from bids_neuropoly import bids
from medicaltorch import datasets as mt_datasets
from medicaltorch.filters import SliceFilter
from ivadomed.utils import *
from ivadomed import __path__

from sklearn.preprocessing import OneHotEncoder
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import numpy as np
import json
from PIL import Image
from glob import glob
from copy import deepcopy
from tqdm import tqdm
import nibabel as nib
import torch

with open(os.path.join(__path__[0], "../config/contrast_dct.json"), "r") as fhandle:
    GENERIC_CONTRAST = json.load(fhandle)
MANUFACTURER_CATEGORY = {'Siemens': 0, 'Philips': 1, 'GE': 2}
CONTRAST_CATEGORY = {"T1w": 0, "T2w": 1, "T2star": 2,
                     "acq-MToff_MTS": 3, "acq-MTon_MTS": 4, "acq-T1w_MTS": 5}
AXIS_DCT = {'sagittal': 0, 'coronal': 1, 'axial': 2}


class Bids3DDataset(mt_datasets.MRI3DSubVolumeSegmentationDataset):
    def __init__(self, root_dir, subject_lst, target_suffix, contrast_lst, contrast_balance={}, slice_axis=2,
                 cache=True, transform=None, metadata_choice=False, canonical=True, labeled=True, roi_suffix=None,
                 multichannel=False, length=(64, 64, 64), padding=0):
        dataset = BidsDataset(root_dir,
                              subject_lst=subject_lst,
                              target_suffix=target_suffix,
                              roi_suffix=roi_suffix,
                              contrast_lst=contrast_lst,
                              metadata_choice=metadata_choice,
                              contrast_balance=contrast_balance,
                              slice_axis=slice_axis,
                              transform=transform,
                              multichannel=multichannel)

        super().__init__(dataset.filename_pairs, cache, length=length, padding=padding, transform=transform,
                         canonical=canonical)


class BidsDataset(mt_datasets.MRI2DSegmentationDataset):
    def __init__(self, root_dir, subject_lst, target_suffix, contrast_lst, contrast_balance={}, slice_axis=2,
                 cache=True, transform=None, metadata_choice=False, slice_filter_fn=None,
                 canonical=True, labeled=True, roi_suffix=None, multichannel=False, missing_modality=False):

        self.bids_ds = bids.BIDS(root_dir)
        self.filename_pairs = []
        if metadata_choice == 'mri_params':
            self.metadata = {"FlipAngle": [], "RepetitionTime": [],
                             "EchoTime": [], "Manufacturer": []}

        bids_subjects = [s for s in self.bids_ds.get_subjects() if s.record["subject_id"] in subject_lst]

        # Create a list with the filenames for all contrasts and subjects
        subjects_tot = []
        for subject in bids_subjects:
            subjects_tot.append(str(subject.record["absolute_path"]))

        # Create a dictionary with the number of subjects for each contrast of contrast_balance

        tot = {contrast: len([s for s in bids_subjects if s.record["modality"] == contrast])
               for contrast in contrast_balance.keys()}

        # Create a counter that helps to balance the contrasts
        c = {contrast: 0 for contrast in contrast_balance.keys()}

        multichannel_subjects = {}
        if multichannel:
            num_contrast = len(contrast_lst)
            idx_dict = {}
            for idx, contrast in enumerate(contrast_lst):
                idx_dict[contrast] = idx
            multichannel_subjects = {subject: {"absolute_paths": [None] * num_contrast,
                                               "deriv_path": None,
                                               "roi_filename": None,
                                               "metadata": [None] * num_contrast} for subject in subject_lst}

        for subject in tqdm(bids_subjects, desc="Loading dataset"):
            if subject.record["modality"] in contrast_lst:

                # Training & Validation: do not consider the contrasts over the threshold contained in contrast_balance
                if subject.record["modality"] in contrast_balance.keys():
                    c[subject.record["modality"]] = c[subject.record["modality"]] + 1
                    if c[subject.record["modality"]] / tot[subject.record["modality"]] > contrast_balance[
                            subject.record["modality"]]:
                        continue

                if not subject.has_derivative("labels"):
                    print("Subject without derivative, skipping.")
                    continue
                derivatives = subject.get_derivatives("labels")
                target_filename, roi_filename = [None] * len(target_suffix), None

                for deriv in derivatives:
                    for idx, suffix in enumerate(target_suffix):
                        if deriv.endswith(subject.record["modality"] + suffix + ".nii.gz"):
                            target_filename[idx] = deriv

                    if not (roi_suffix is None) and\
                            deriv.endswith(subject.record["modality"] + roi_suffix + ".nii.gz"):
                        roi_filename = [deriv]

                if (not any(target_filename)) or (not (roi_suffix is None) and (roi_filename is None)):
                    continue

                if not subject.has_metadata():
                    print("Subject without metadata.")
                    metadata = {}
                else:
                    metadata = subject.metadata()

                # add contrast to metadata
                metadata['contrast'] = subject.record["modality"]

                if metadata_choice == 'mri_params':
                    def _check_isMRIparam(mri_param_type, mri_param):
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

                            self.metadata[mri_param_type].append(value)
                            return True

                    if not all([_check_isMRIparam(m, metadata) for m in self.metadata.keys()]):
                        continue

                # Fill multichannel dictionary
                if multichannel:
                    idx = idx_dict[subject.record["modality"]]
                    subj_id = subject.record["subject_id"]
                    multichannel_subjects[subj_id]["absolute_paths"][idx] = subject.record.absolute_path
                    multichannel_subjects[subj_id]["deriv_path"] = target_filename
                    multichannel_subjects[subj_id]["metadata"][idx] = metadata
                    if roi_filename:
                        multichannel_subjects[subj_id]["roi_filename"] = roi_filename

                else:
                    self.filename_pairs.append(([subject.record.absolute_path],
                                                target_filename, roi_filename, [metadata]))

        if multichannel:
            for subject in multichannel_subjects.values():
                if not None in subject["absolute_paths"]:
                    self.filename_pairs.append((subject["absolute_paths"], subject["deriv_path"],
                                                subject["roi_filename"], subject["metadata"]))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)


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

    def _get_label(self, dataset, idx):
        sample_gt = np.array(dataset[idx]['gt'])
        if np.any(sample_gt):
            return 1
        else:
            return 0

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.nb_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def load_dataset(data_list, data_transform, context):
    if context["unet_3D"]:
        dataset = Bids3DDataset(context["bids_path"],
                                subject_lst=data_list,
                                target_suffix=context["target_suffix"],
                                roi_suffix=context["roi_suffix"],
                                contrast_lst=context["contrast_train_validation"],
                                metadata_choice=context["metadata"],
                                contrast_balance=context["contrast_balance"],
                                slice_axis=AXIS_DCT[context["slice_axis"]],
                                transform=data_transform,
                                multichannel=context['multichannel'],
                                length=context["length_3D"],
                                padding=context["padding_3D"])
    else:
        dataset = BidsDataset(context["bids_path"],
                              subject_lst=data_list,
                              target_suffix=context["target_suffix"],
                              roi_suffix=context["roi_suffix"],
                              contrast_lst=context["contrast_train_validation"],
                              metadata_choice=context["metadata"],
                              contrast_balance=context["contrast_balance"],
                              slice_axis=AXIS_DCT[context["slice_axis"]],
                              transform=data_transform,
                              multichannel=context['multichannel'],
                              slice_filter_fn=SliceFilter(**context["slice_filter"]),
                              missing_modality=context['missing_modality'])
    return dataset

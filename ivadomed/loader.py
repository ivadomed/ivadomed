from bids_neuropoly import bids
from medicaltorch import datasets as mt_datasets

from sklearn.preprocessing import OneHotEncoder
from scipy.signal import argrelextrema
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import numpy as np
import json
from glob import glob
from copy import deepcopy
from tqdm import tqdm
import nibabel as nib
from PIL import Image
import torch

with open("config/contrast_dct.json", "r") as fhandle:
    GENERIC_CONTRAST = json.load(fhandle)
MANUFACTURER_CATEGORY = {'Siemens': 0, 'Philips': 1, 'GE': 2}
CONTRAST_CATEGORY = {"T1w": 0, "T2w": 1, "T2star": 2, "acq-MToff_MTS": 3, "acq-MTon_MTS": 4, "acq-T1w_MTS": 5}


class SegmentationPair2D(object):
    """This class is used to build 2D segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).

    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """
    def __init__(self, input_filename, gt_filename, cache=True, canonical=False):

        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache

        self.input_handle = []
        for input_file in self.input_filename:
            input = nib.load(input_file)
            self.input_handle.append(input)
            if len(input.shape) > 3:
                raise RuntimeError("4-dimensional volumes not supported.")

        # Unlabeled data (inference time)

        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)


        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        if self.canonical:
            for idx, handle in enumerate(self.input_handle):
                self.input_handle[idx] = nib.as_closest_canonical(handle)

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)


    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = []
        for handle in self.input_handle:
            input_shape.append(handle.header.get_data_shape())

            if not len(set(input_shape)):
                raise RuntimeError('Inputs have different dimensions.')

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape[0], gt_shape

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'

        input_data = []
        for handle in self.input_handle:
            input_data.append(handle.get_fdata(cache_mode, dtype=np.float32))

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        return input_data, gt_data

    def get_pair_slice(self, slice_index, slice_axis=2):
        """Return the specified slice from (input, ground truth).

        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        if self.cache:
            input_dataobj, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = []
            input_dataobj.append(handle.dataobj for handle in self.input_handle)

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        input_slice = []
        for data_object in input_dataobj:
            if slice_axis == 2:
                input_slice.append(np.asarray(data_object[..., slice_index],
                                         dtype=np.float32))
            elif slice_axis == 1:
                input_slice.append(np.asarray(data_object[:, slice_index, ...],
                                         dtype=np.float32))
            elif slice_axis == 0:
                input_slice.append(np.asarray(data_object[slice_index, ...],
                                         dtype=np.float32))

        # Handle the case for unlabeled data
        gt_meta_dict = None
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

            gt_meta_dict = mt_datasets.SampleMetadata({
                "zooms": self.gt_handle.header.get_zooms()[:2],
                "data_shape": self.gt_handle.header.get_data_shape()[:2],
            })

        input_meta_dict = []
        for handle in self.input_handle:
            input_meta_dict.append(mt_datasets.SampleMetadata({
                "zooms": handle.header.get_zooms()[:2],
                "data_shape": handle.header.get_data_shape()[:2],
            }))

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }

        return dreturn

class BIDSSegPair2D(SegmentationPair2D):
    def __init__(self, input_filename, gt_filename, metadata, contrast, cache=True, canonical=True):
        super().__init__(input_filename, gt_filename, canonical=canonical)

        self.metadata = []
        for data in metadata:
            data["input_filename"] = input_filename
            data["gt_filename"] = gt_filename
            data["contrast"] = contrast  # eg T2w
            self.metadata.append(data)

    def get_pair_slice(self, slice_index, slice_axis=2):
        dreturn = super().get_pair_slice(slice_index, slice_axis)

        for idx, metadata in enumerate(self.metadata):
            metadata["slice_index"] = slice_index
            self.metadata[idx] = metadata
            dreturn["input_metadata"][idx]["bids_metadata"] = metadata
        return dreturn


class MRI2DBidsSegDataset(mt_datasets.MRI2DSegmentationDataset):
    def __init__(self, filename_pairs, slice_axis=2, cache=True,
                 transform=None, slice_filter_fn=None, canonical=False, multichannel=False):
        self.multichannel = multichannel
        super().__init__(filename_pairs, slice_axis=slice_axis, cache=cache, transform=transform, slice_filter_fn=slice_filter_fn,
                         canonical=canonical)

    def _load_filenames(self):
        for input_filename, gt_filename, bids_metadata, contrast in self.filename_pairs:
            segpair = BIDSSegPair2D(input_filename, gt_filename,
                                    bids_metadata, contrast)
            self.handlers.append(segpair)

    def _prepare_indexes(self):
        for segpair in self.handlers:
            input_data_shape, _ = segpair.get_pair_shapes()
            for segpair_slice in range(input_data_shape[self.slice_axis]):

                # Check if slice pair should be used or not
                if self.slice_filter_fn:
                    slice_pair = segpair.get_pair_slice(segpair_slice,
                                                        self.slice_axis)

                    for slice in slice_pair['input']:
                        single_slice_pair = slice_pair
                        single_slice_pair['input'] = slice
                        filter_fn_ret = self.slice_filter_fn(slice_pair)

                    if not filter_fn_ret:
                        continue

                item = (segpair, segpair_slice)
                self.indexes.append(item)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).

        :param index: slice index.
        """
        segpair, segpair_slice = self.indexes[index]
        pair_slice = segpair.get_pair_slice(segpair_slice,
                                            self.slice_axis)

        input_tensors = []
        input_metadata = []
        data_dict = {}
        for idx, input_slice in enumerate(pair_slice["input"]):
            # Consistency with torchvision, returning PIL Image
            # Using the "Float mode" of PIL, the only mode
            # supporting unbounded float32 values
            input_img = Image.fromarray(input_slice, mode='F')

            # Handle unlabeled data
            if pair_slice["gt"] is None:
                gt_img = None
            else:
                gt_img = Image.fromarray(pair_slice["gt"], mode='F')

            data_dict = {
                'input': input_img,
                'gt': gt_img,
                'input_metadata': pair_slice['input_metadata'][idx],
                'gt_metadata': pair_slice['gt_metadata'],
            }

            if self.transform is not None:
                data_dict = self.transform(data_dict)
            input_tensors.append(data_dict['input'])
            input_metadata.append(data_dict['input_metadata'])

        if len(input_tensors) > 1:
            data_dict['input'] = torch.squeeze(torch.stack(input_tensors, dim=0))
            data_dict['input_metadata'] = input_metadata

        return data_dict

class BidsDataset(MRI2DBidsSegDataset):
    def __init__(self, root_dir, subject_lst, gt_suffix, contrast_lst, contrast_balance={}, slice_axis=2, cache=True,
                 transform=None, metadata_choice=False, slice_filter_fn=None,
                 canonical=True, labeled=True, multichannel=False):

        self.bids_ds = bids.BIDS(root_dir)
        self.filename_pairs = []
        if metadata_choice == 'mri_params':
            self.metadata = {"FlipAngle": [], "RepetitionTime": [], "EchoTime": [], "Manufacturer": []}

        bids_subjects = [s for s in self.bids_ds.get_subjects() if s.record["subject_id"] in subject_lst]

        # Create a list with the filenames for all contrasts and subjects
        subjects_tot = []
        for subject in bids_subjects:
            subjects_tot.append(str(subject.record["absolute_path"]))

        # Create a dictionary with the number of subjects for each contrast of contrast_balance
        tot = {contrast: len([s for s in bids_subjects if s.record["modality"] == contrast]) for contrast in contrast_balance.keys()}
        # Create a counter that helps to balance the contrasts
        c = {contrast: 0 for contrast in contrast_balance.keys()}

        multichannel_subjects = {}
        if multichannel:
            multichannel_subjects = {subject: {"absolute_paths": [],
                                                  "deriv_path": None,
                                                  "metadata": [],
                                                  "modalities": []} for subject in subject_lst}

        for subject in tqdm(bids_subjects, desc="Loading dataset"):
            if subject.record["modality"] in contrast_lst:

                # Training & Validation: do not consider the contrasts over the threshold contained in contrast_balance
                if subject.record["modality"] in contrast_balance.keys():
                    c[subject.record["modality"]] = c[subject.record["modality"]] + 1
                    if c[subject.record["modality"]] / tot[subject.record["modality"]] > contrast_balance[subject.record["modality"]]:
                        continue

                if not subject.has_derivative("labels"):
                    print("Subject without derivative, skipping.")
                    continue
                derivatives = subject.get_derivatives("labels")
                cord_label_filename = None

                for deriv in derivatives:
                    if deriv.endswith(subject.record["modality"]+gt_suffix+".nii.gz"):
                        cord_label_filename = deriv

                if cord_label_filename is None:
                    continue

                if not subject.has_metadata():
                    print("Subject without metadata.")
                    continue

                metadata = subject.metadata()
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
                                    value = np.mean([float(v) for v in mri_param[mri_param_type].split(',')])

                            self.metadata[mri_param_type].append(value)
                            return True

                    if not all([_check_isMRIparam(m, metadata) for m in self.metadata.keys()]):
                        continue

                # Fill multichannel dictionary
                if multichannel:
                    id = subject.record["subject_id"]
                    multichannel_subjects[id]["absolute_paths"].append(subject.record.absolute_path)
                    multichannel_subjects[id]["deriv_path"] = cord_label_filename
                    multichannel_subjects[id]["metadata"].append(subject.metadata())
                    multichannel_subjects[id]["modalities"].append(subject.record["modality"])

                else:
                    self.filename_pairs.append(([subject.record.absolute_path],
                                                cord_label_filename, [metadata], [subject.record["modality"]]))

        if multichannel:
            for subject in multichannel_subjects.values():
                if len(subject["absolute_paths"]):
                    self.filename_pairs.append((subject["absolute_paths"], subject["deriv_path"], subject["metadata"], subject["modalities"]))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)


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
        X_train, X_remain = train_test_split(df['participant_id'].tolist(), train_size=train_frac, random_state=random_seed)
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
                v = subject["input_metadata"]["bids_metadata"][m]
                p = clustering_models[m].predict(v)
                s_out["input_metadata"]["bids_metadata"][m] = p
                if debugging:
                    print("{}: {} --> {}".format(m, v, p))

            # categorize manufacturer info based on the MANUFACTURER_CATEGORY dictionary
            manufacturer = subject["input_metadata"]["bids_metadata"]["Manufacturer"]
            if manufacturer in MANUFACTURER_CATEGORY:
                s_out["input_metadata"]["bids_metadata"]["Manufacturer"] = MANUFACTURER_CATEGORY[manufacturer]
                if debugging:
                    print("Manufacturer: {} --> {}".format(manufacturer, MANUFACTURER_CATEGORY[manufacturer]))
            else:
                print("{} with unknown manufacturer.".format(subject))
                s_out["input_metadata"]["bids_metadata"]["Manufacturer"] = -1  # if unknown manufacturer, then value set to -1

            s_out["input_metadata"]["bids_metadata"] = [s_out["input_metadata"]["bids_metadata"][k] for k in
                                                        ["FlipAngle", "RepetitionTime", "EchoTime", "Manufacturer"]]
        else:
            generic_contrast = GENERIC_CONTRAST[subject["input_metadata"]["bids_metadata"]["contrast"][0]] # FILM is only single channel
            label_contrast = CONTRAST_CATEGORY[generic_contrast]
            s_out["input_metadata"]["bids_metadata"] = [label_contrast]

        s_out["input_metadata"]["contrast"] = subject["input_metadata"]["bids_metadata"]["contrast"]

        if train_set:
            X_train_ohe.append(s_out["input_metadata"]["bids_metadata"])
        ds_out.append(s_out)

        del s_out, subject

    if train_set:
        X_train_ohe = np.vstack(X_train_ohe)
        ohe.fit(X_train_ohe)
        return ds_out, ohe
    else:
        return ds_out

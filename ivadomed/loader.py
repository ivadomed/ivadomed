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

with open("config/contrast_dct.json", "r") as fhandle:
    GENERIC_CONTRAST = json.load(fhandle)
MANUFACTURER_CATEGORY = {'Siemens': 0, 'Philips': 1, 'GE': 2}
CONTRAST_CATEGORY = {"T1w": 0, "T2w": 1, "T2star": 2, "acq-MToff_MTS": 3, "acq-MTon_MTS": 4, "acq-T1w_MTS": 5}

class BIDSSegPair2D(mt_datasets.SegmentationPair2D):
    def __init__(self, input_filename, gt_filename, metadata, contrast, cache=True, canonical=True):
        super().__init__(input_filename, gt_filename, canonical=canonical)

        self.metadata = metadata
        self.metadata["input_filename"] = input_filename
        self.metadata["gt_filename"] = gt_filename
        self.metadata["contrast"] = contrast  # eg T2w

    def get_pair_slice(self, slice_index, slice_axis=2):
        dreturn = super().get_pair_slice(slice_index, slice_axis)
        self.metadata["slice_index"] = slice_index
        dreturn["input_metadata"]["bids_metadata"] = self.metadata
        return dreturn


class MRI2DBidsSegDataset(mt_datasets.MRI2DSegmentationDataset):
    def _load_filenames(self):
        for input_filename, target_filename, bids_metadata, contrast, roi_filename in self.filename_pairs:
            segpair = BIDSSegPair2D(input_filename, target_filename,
                                    bids_metadata, contrast, roi_filename)
            roipair = BIDSSegPair2D(input_filename, roi_filename,
                                    bids_metadata, contrast)

            self.handlers.append([segpair, roipair])

    def _prepare_indexes(self):
        for seg_roi_pairs in self.handlers:
            input_data_shape, _ = seg_roi_pairs[0].get_pair_shapes()
            for idx_pair_slice in range(input_data_shape[self.slice_axis]):

                # Check if slice pair should be used or not
                if self.slice_filter_fn:
                    slice_pair_seg = seg_roi_pairs[0].get_pair_slice(idx_pair_slice,
                                                            self.slice_axis)
                    slice_pair_roi = seg_roi_pairs[1].get_pair_slice(idx_pair_slice,
                                                            self.slice_axis)
                    filter_fn_ret_seg = self.slice_filter_fn(slice_pair_seg)
                    filter_fn_ret_roi = self.slice_filter_fn(slice_pair_roi)
                    if (not filter_fn_ret_seg) or (not filter_fn_ret_roi):
                        continue

                item = (seg_roi_pairs[0], seg_roi_pairs[1], idx_pair_slice)
                self.indexes.append(item)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, target),
        or (input, target, roi).
        :param index: slice index.
        """
        segpair, roipair, pair_slice = self.indexes[index]
        seg_pair_slice = segpair.get_pair_slice(pair_slice,
                                                self.slice_axis)
        roi_pair_slice = roipair.get_pair_slice(pair_slice,
                                                self.slice_axis)

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values
        input_img = Image.fromarray(seg_pair_slice["input"], mode='F')

        # Handle unlabeled data
        if seg_pair_slice["gt"] is None:
            gt_img = None
        else:
            gt_img = Image.fromarray(seg_pair_slice["gt"], mode='F')

        if roi_pair_slice["gt"] is None:
            roi_img = None
        else:
            roi_img = Image.fromarray(roi_pair_slice["gt"], mode='F')

        data_dict = {
            'input': input_img,
            'target': gt_img,
            'roi': roi_img,
            'input_metadata': seg_pair_slice['input_metadata'],
            'target_metadata': seg_pair_slice['gt_metadata'],
            'roi_metadata': roi_pair_slice['gt_metadata'],
        }

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict


class BidsDataset(MRI2DBidsSegDataset):
    def __init__(self, root_dir, subject_lst, target_suffix, contrast_lst, contrast_balance={}, slice_axis=2, cache=True,
                 transform=None, metadata_choice=False, slice_filter_fn=None,
                 canonical=True, labeled=True, roi_suffix=None):

        self.bids_ds = bids.BIDS(root_dir)
        self.filename_pairs = []
        if metadata_choice == 'mri_params':
            self.metadata = {"FlipAngle": [], "RepetitionTime": [], "EchoTime": [], "Manufacturer": []}

        # Selecting subjects from Training / Validation / Testing
        bids_subjects = [s for s in self.bids_ds.get_subjects() if s.record["subject_id"] in subject_lst]

        # Create a list with the filenames for all contrasts and subjects
        subjects_tot = []
        for subject in bids_subjects:
            subjects_tot.append(str(subject.record["absolute_path"]))

        # Create a dictionary with the number of subjects for each contrast of contrast_balance
        tot = {contrast: len([s for s in bids_subjects if s.record["modality"] == contrast]) for contrast in contrast_balance.keys()}
        # Create a counter that helps to balance the contrasts
        c = {contrast: 0 for contrast in contrast_balance.keys()}

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
                target_filename, roi_filename = None, None

                for deriv in derivatives:
                    if deriv.endswith(subject.record["modality"]+target_suffix+".nii.gz"):
                        target_filename = deriv
                    if not (roi_suffix is None) and deriv.endswith(subject.record["modality"]+roi_suffix+".nii.gz"):
                        roi_filename = deriv

                if (target_filename is None) or (not (roi_suffix is None) and (roi_filename is None)):
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

                self.filename_pairs.append((subject.record.absolute_path,
                                            target_filename, metadata, subject.record["modality"],
                                            roi_filename))

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
            generic_contrast = GENERIC_CONTRAST[subject["input_metadata"]["bids_metadata"]["contrast"]]
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

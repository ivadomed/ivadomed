from bids_neuropoly import bids
from medicaltorch import datasets as mt_datasets

from sklearn.preprocessing import OneHotEncoder
from scipy.signal import argrelextrema
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import numpy as np
from glob import glob
from copy import deepcopy
from tqdm import tqdm

MANUFACTURER_CATEGORY = {'Siemens': 0, 'Philips': 1, 'GE': 2}


class BIDSSegPair2D(mt_datasets.SegmentationPair2D):
    def __init__(self, input_filename, gt_filename, metadata, contrast):
        super().__init__(input_filename, gt_filename)
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
        for input_filename, gt_filename, bids_metadata, contrast in self.filename_pairs:
            segpair = BIDSSegPair2D(input_filename, gt_filename,
                                    bids_metadata, contrast)
            self.handlers.append(segpair)


class BidsDataset(MRI2DBidsSegDataset):
    def __init__(self, root_dir, subject_lst, contrast_lst, contrast_balance={}, slice_axis=2, cache=True,
                 transform=None, slice_filter_fn=None,
                 canonical=False, labeled=True):

        self.bids_ds = bids.BIDS(root_dir)
        self.filename_pairs = []
        self.metadata = {"FlipAngle": [], "RepetitionTime": [], "EchoTime": [], "Manufacturer": []}

        # Selecting subjects from Training / Validation / Testing
        bids_subjects = [s for s in self.bids_ds.get_subjects() if s.record["subject_id"] in subject_lst]

        # Create a dictionary with the number of subjects for each contrast
        tot = {subject.record["modality"]: len([s for s in bids_subjects if str(subject.record["modality"]) in s]) for subject in tqdm(bids_subjects, desc="Loading dataset")}
        print("Number of subjects per contrast: {}".format(tot))

        # Create a counter that helps to balance the contrasts
        c = {subject.record["modality"]: 0 for subject in tqdm(bids_subjects, desc="Loading dataset")}
        print("Counter per contrast: {}".format(c))

        for subject in tqdm(bids_subjects, desc="Loading dataset"):
            if subject.record["modality"] in contrast_lst:

                # Training & Validation: do not consider the contrasts over the threshold contained in contrast_balance
                if subject.record["modality"] in contrast_balance.keys():
                    c[subject.record["modality"]] = c[subject.record["modality"]] + 1
                    if c[subject.record["modality"]] / tot[subject.record["modality"]] > contrast_balance[subject.record["modality"]]:
                        print("{} from {}, skipped because over contrast threshold."
                              .format(subject.record["modality"], subject))
                        continue

                if not subject.has_derivative("labels"):
                    print("Subject without derivative, skipping.")
                    continue
                derivatives = subject.get_derivatives("labels")
                cord_label_filename = None

                for deriv in derivatives:
                    if deriv.endswith("seg-manual.nii.gz"):
                        cord_label_filename = deriv

                if cord_label_filename is None:
                    continue

                if not subject.has_metadata():
                    print("Subject without metadata.")
                    continue

                def _check_isMetadata(metadata_type, metadata):
                    if metadata_type not in metadata:
                        print("{} without {}, skipping.".format(subject, metadata_type))
                        return False
                    else:
                        if metadata_type == "Manufacturer":
                            value = metadata[metadata_type]
                        else:
                            if isinstance(metadata[metadata_type], (int, float)):
                                value = float(metadata[metadata_type])
                            else:  # eg multi-echo data have 3 echo times
                                value = np.mean([float(v) for v in metadata[metadata_type].split(',')])

                        self.metadata[metadata_type].append(value)
                        return True

                metadata = subject.metadata()
                if not all([_check_isMetadata(m, metadata) for m in self.metadata.keys()]):
                    continue

                self.filename_pairs.append((subject.record.absolute_path,
                                            cord_label_filename, metadata, subject.record["modality"]))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)


def split_dataset(path_folder, center_test_lst, random_seed, train_frac=0.8):
    # read participants.tsv as pandas dataframe
    df = bids.BIDS(path_folder).participants.content

    # make sure that subjects coming from some centers are unseen during training
    X_test = df[df['institution_id'].isin(center_test_lst)]['participant_id'].tolist()
    X_remain = df[~df['institution_id'].isin(center_test_lst)]['participant_id'].tolist()

    # split using sklearn function
    X_train, X_tmp = train_test_split(X_remain, train_size=train_frac, random_state=random_seed)
    if len(X_test):  # X_test contains data from centers unseen during the training, eg SpineGeneric
        X_val = X_tmp
    else:  # X_test contains data from centers seen during the training, eg gm_challenge
        X_val, X_test = train_test_split(X_tmp, train_size=0.5, random_state=random_seed)

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


def normalize_metadata(ds_in, clustering_models, debugging, train_set=False):
    if train_set:
        # Initialise One Hot Encoder
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_train_ohe = []

    ds_out = []
    for idx, subject in enumerate(ds_in):
        s_out = deepcopy(subject)

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
            s_out["input_metadata"]["bids_metadata"][
                "Manufacturer"] = -1  # if unknown manufacturer, then value set to -1

        s_out["input_metadata"]["bids_metadata"] = [s_out["input_metadata"]["bids_metadata"][k] for k in
                                                    ["FlipAngle", "RepetitionTime", "EchoTime", "Manufacturer"]]

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

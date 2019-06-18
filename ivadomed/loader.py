from bids_neuropoly import bids
from medicaltorch import datasets as mt_datasets

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import MeanShift, estimate_bandwidth

import numpy as np
from copy import deepcopy


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
    def __init__(self, root_dir, contrast_lst, slice_axis=2, cache=True,
                 transform=None, slice_filter_fn=None,
                 canonical=False, labeled=True):

        self.bids_ds = bids.BIDS(root_dir)
        self.filename_pairs = []
        self.metadata = {"FlipAngle": [], "RepetitionTime": [], "EchoTime": [], "Manufacturer": []}

        for subject in self.bids_ds.get_subjects():
            if subject.record["modality"] in contrast_lst:

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
                        value = metadata[metadata_type] if metadata_type == "Manufacturer" else float(metadata[metadata_type])
                        self.metadata[metadata_type].append(value)
                        return True

                metadata = subject.metadata()
                if not all([_check_isMetadata(m, metadata) for m in self.metadata.keys()]):
                    continue

                self.filename_pairs.append((subject.record.absolute_path,
                                            cord_label_filename, metadata, subject.record["modality"]))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)


class Kde_model():
    def __init__(self):
        self.kde = KernelDensity()
        self.minima = None

    def train(self, data, value_range, gridsearch_bandwidth_range):
        # reshape data to fit sklearn
        data = np.array(data).reshape(-1,1)

        # use grid search cross-validation to optimize the bandwidth
        params = {'bandwidth': gridsearch_bandwidth_range}
        grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
        grid.fit(data)

        # use the best estimator to compute the kernel density estimate
        self.kde = grid.best_estimator_

        # fit kde with the best bandwidth
        self.kde.fit(data)

        s = value_range
        e = self.kde.score_samples(s.reshape(-1,1))

        # find local minima
        self.minima = s[argrelextrema(e, np.less)[0]]

    def predict(self, data):
        # len - 1
        class_lst = []
        for d in data[:,0]:
            x = [i for i, m in enumerate(self.minima) if d < m]
            class_cur = min(x) if len(x) else len(self.minima)

            class_lst.append(class_cur)

        return np.array(class_lst).reshape(-1,1)


def clustering_fit(datasets, key_lst):
    """This function creates clustering models for each metadata type,
    using MeanShift algorithm.

    :param datasets (list of lists): data for each dataset
    :param key_lst (list of strings): names of metadata to cluster
    :return: clustering model for each metadata type
    """
    model_dct, encoder_dct = {}, {}
    for k in key_lst:
        k_data = [value for dataset in datasets for value in dataset[k]]

        X = np.array(list(zip(k_data, np.zeros(len(k_data)))))  # format the data before sending to the clustering algo
        bandwidth = estimate_bandwidth(X, quantile=0.1)  # estimate the bandwidth to use with the mean-shift algo
        clf = MeanShift(bandwidth=bandwidth if bandwidth > 0.0 else None, bin_seeding=True)  # mean shift clustering using a flat kernel
        clf.fit(X)
        model_dct[k] = clf
        del clf

    return model_dct


def normalize_metadata(ds_lst_in, clustering_models, debugging, train_set=False):

    if train_set:
        # Initialise One Hot Encoder
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_train_ohe = []

    ds_lst_out = []
    for ds_in in ds_lst_in:
        ds_out = []
        for idx, subject in enumerate(ds_in):
            s_out = deepcopy(subject)

            # categorize flip angle value using meanShift
            flip_angle = [subject["input_metadata"]["bids_metadata"]["FlipAngle"]]
            int_value = clustering_models["FlipAngle"].predict(np.array(list(zip(flip_angle, np.zeros(1)))))
            s_out["input_metadata"]["bids_metadata"]["FlipAngle"] = int_value[0]

            # categorize repetition time value using meanShift
            repetition_time = [subject["input_metadata"]["bids_metadata"]["RepetitionTime"]]
            int_value = clustering_models["RepetitionTime"].predict(np.array(list(zip(repetition_time, np.zeros(1)))))
            s_out["input_metadata"]["bids_metadata"]["RepetitionTime"] = int_value[0]

            # categorize echo time value using meanShift
            echo_time = [subject["input_metadata"]["bids_metadata"]["EchoTime"]]
            int_value = clustering_models["EchoTime"].predict(np.array(list(zip(echo_time, np.zeros(1)))))
            s_out["input_metadata"]["bids_metadata"]["EchoTime"] = int_value[0]

            # categorize manufacturer info based on the MANUFACTURER_CATEGORY dictionary
            manufacturer = subject["input_metadata"]["bids_metadata"]["Manufacturer"]
            if manufacturer in MANUFACTURER_CATEGORY:
                s_out["input_metadata"]["bids_metadata"]["Manufacturer"] = MANUFACTURER_CATEGORY[manufacturer]
            else:
                print("{} with unknown manufacturer.".format(subject))
                s_out["input_metadata"]["bids_metadata"]["Manufacturer"] = -1  # if unknown manufacturer, then value set to -1

            if debugging:
                print("\nFlip Angle: {} --> {}".format(flip_angle[0], s_out["input_metadata"]["bids_metadata"]["FlipAngle"]))
                print("Repetition Time: {} --> {}".format(repetition_time[0], s_out["input_metadata"]["bids_metadata"]["RepetitionTime"]))
                print("Echo Time: {} --> {}".format(echo_time[0], s_out["input_metadata"]["bids_metadata"]["EchoTime"]))
                print("Manufacturer: {} --> {}".format(manufacturer, s_out["input_metadata"]["bids_metadata"]["Manufacturer"]))

            s_out["input_metadata"]["bids_metadata"] = [s_out["input_metadata"]["bids_metadata"][k] for k in ["FlipAngle", "RepetitionTime", "EchoTime", "Manufacturer"]]

            s_out["input_metadata"]["contrast"] = subject["input_metadata"]["bids_metadata"]["contrast"]

            if train_set:
                X_train_ohe.append(s_out["input_metadata"]["bids_metadata"])
            ds_out.append(s_out)

            del s_out, subject

        ds_lst_out.append(ds_out)

    if train_set:
        X_train_ohe = np.vstack(X_train_ohe)
        ohe.fit(X_train_ohe)
        return ds_lst_out, ohe
    else:
        return ds_lst_out

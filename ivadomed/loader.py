from bids_neuropoly import bids
from medicaltorch import datasets as mt_datasets

from sklearn.cluster import MeanShift, estimate_bandwidth

import numpy as np
from copy import deepcopy


MANUFACTURER_CATEGORY = {'Siemens': 0, 'Philips': 1, 'GE': 2}


class BIDSSegPair2D(mt_datasets.SegmentationPair2D):
    def __init__(self, input_filename, gt_filename, metadata):
        super().__init__(input_filename, gt_filename)
        self.metadata = metadata
        self.metadata["input_filename"] = input_filename
        self.metadata["gt_filename"] = gt_filename

    def get_pair_slice(self, slice_index, slice_axis=2):
        dreturn = super().get_pair_slice(slice_index, slice_axis)
        self.metadata["slice_index"] = slice_index
        dreturn["input_metadata"]["bids_metadata"] = self.metadata
        return dreturn


class MRI2DBidsSegDataset(mt_datasets.MRI2DSegmentationDataset):
    def _load_filenames(self):
        for input_filename, gt_filename, bids_metadata in self.filename_pairs:
            segpair = BIDSSegPair2D(input_filename, gt_filename,
                                    bids_metadata)
            self.handlers.append(segpair)


class BidsDataset(MRI2DBidsSegDataset):
    def __init__(self, root_dir, slice_axis=2, cache=True,
                 transform=None, slice_filter_fn=None,
                 canonical=False, labeled=True):
        print(root_dir)
        self.bids_ds = bids.BIDS(root_dir)
        self.filename_pairs = []
        self.metadata = {"FlipAngle": [], "RepetitionTime": [], "EchoTime": [], "Manufacturer": []}

        for subject in self.bids_ds.get_subjects():
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

            metadata = subject.metadata()
            if "FlipAngle" not in metadata:
                print("{} without FlipAngle, skipping.".format(subject))
                continue
            else:
                self.metadata["FlipAngle"].append(float(metadata["FlipAngle"]))

            if "EchoTime" not in metadata:
                print("{} without EchoTime, skipping.".format(subject))
                continue
            else:
                self.metadata["EchoTime"].append(float(metadata["EchoTime"]))

            if "RepetitionTime" not in metadata:
                print("{} without RepetitionTime, skipping.".format(subject))
                continue
            else:
                self.metadata["RepetitionTime"].append(float(metadata["RepetitionTime"]))

            if "Manufacturer" not in metadata:
                print("{} without Manufacturer, skipping.".format(subject))
                continue
            else:
                self.metadata["Manufacturer"].append(metadata["Manufacturer"])

            self.filename_pairs.append((subject.record.absolute_path,
                                        cord_label_filename, metadata))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)


def _rescale_value(value_in, range_in, range_out):
    delta_in = range_in[1] - range_in[0]
    delta_out = range_out[1] - range_out[0]
    return (delta_out * (value_in - range_in[0]) / delta_in) + range_out[0]


def clustering_fit(datasets, key_lst):
    model_dct = {}
    for k in key_lst:
        k_data = [value for dataset in datasets for value in dataset[k]]
        X = np.array(list(zip(k_data, np.zeros(len(k_data)))))  # format the data before sending to the clustering algo
        bandwidth = estimate_bandwidth(X, quantile=0.1)  # estimate the bandwidth to use with the mean-shift algo
        clf = MeanShift(bandwidth=bandwidth, bin_seeding=True)  # mean shift clustering using a flat kernel
        clf.fit(X)
        model_dct[k] = clf
        del clf
    return model_dct


def normalize_metadata(ds_lst_in, clustering_models, debugging):
    ds_lst_out = []
    for ds_in in ds_lst_in:
        ds_out = []
        for idx, subject in enumerate(ds_in):
            s_out = deepcopy(subject)

            # categorize flip angle value using meanShift
            flip_angle = [subject["input_metadata"]["bids_metadata"]["FlipAngle"]]
            s_out["input_metadata"]["bids_metadata"]["FlipAngle"] = clustering_models["FlipAngle"].predict(np.array(list(zip(flip_angle, np.zeros(1)))))[0]

            # categorize repetition time value using meanShift
            repetition_time = [subject["input_metadata"]["bids_metadata"]["RepetitionTime"]]
            s_out["input_metadata"]["bids_metadata"]["RepetitionTime"] = clustering_models["RepetitionTime"].predict(np.array(list(zip(repetition_time, np.zeros(1)))))[0]

            # categorize echo time value using meanShift
            echo_time = [subject["input_metadata"]["bids_metadata"]["EchoTime"]]
            s_out["input_metadata"]["bids_metadata"]["EchoTime"] = clustering_models["EchoTime"].predict(np.array(list(zip(echo_time, np.zeros(1)))))[0]

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
            ds_out.append(s_out)

            del s_out, subject

        ds_lst_out.append(ds_out)

    return ds_lst_out

from bids_neuropoly import bids
from medicaltorch import datasets as mt_datasets
from sklearn.cluster import MeanShift, estimate_bandwidth

class BIDSSegPair2D(mt_datasets.SegmentationPair2D):
    def __init__(self, input_filename, gt_filename, metadata):
        super().__init__(input_filename, gt_filename)
        self.metadata = metadata

    def get_pair_slice(self, slice_index, slice_axis=2):
        dreturn = super().get_pair_slice(slice_index, slice_axis)
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
        self.bids_ds = bids.BIDS(root_dir)
        self.filename_pairs = []
        self.metadata = {"FlipAngle": [], "RepetitionTime": [], "EchoTime": []}

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
                print("Subject without cord label.")
                continue

            if not subject.has_metadata():
                print("Subject without metadata.")
                continue

            metadata = subject.metadata()
            if "FlipAngle" not in metadata:
                print("{} without FlipAngle, skipping.".format(subject))
                continue
            else:
                self.metadata["FlipAngle"].append(metadata["FlipAngle"])

            if "EchoTime" not in metadata:
                print("{} without EchoTime, skipping.".format(subject))
                continue
            else:
                self.metadata["EchoTime"].append(metadata["EchoTime"])

            if "RepetitionTime" not in metadata:
                print("{} without RepetitionTime, skipping.".format(subject))
                continue
            else:
                self.metadata["RepetitionTime"].append(metadata["RepetitionTime"])

            self.filename_pairs.append((subject.record.absolute_path,
                                        cord_label_filename, metadata))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)

    def rescale_value(value_in, range_in, range_out):
        delta_in = range_in[1] - range_in[0]
        delta_out = range_out[1] - range_out[0]
        return (delta_out * (value_in - range_in[0]) / delta_in) + range_out[0]

    def fit_cluster(x):
        X = np.array(zip(x, np.zeros(len(x))))
        bandwidth = estimate_bandwidth(X, quantile=0.1)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        return ms

    def normalize_metadata(self, cluster_models=None):
        repetitionTime_values = self.metadata["RepetitionTime"]
        echoTime_values = self.metadata["EchoTime"]

        if cluster_models is None:
            repetitionTime_clusterModel = fit_cluster(repetitionTime_values)
            echoTime_clusterModel = fit_cluster(echoTime_values)
        else:
            repetitionTime_clusterModel = cluster_models["RepetitionTime"]
            echoTime_clusterModel = cluster_models["EchoTime"]

        for index, item in enumerate(self.filename_pairs):
            flip_angle = self.filename_pairs[index][2]["FlipAngle"]
            self.filename_pairs[index][2]["FlipAngle"] = rescale_value(value_in=flip_angle,
                                                                        range_in=[0.0, 360.0],
                                                                        range_out=[0.0, 90.0])

            repetition_time = [self.filename_pairs[index][2]["RepetitionTime"]]
            self.filename_pairs[index][2]["RepetitionTime"] = repetitionTime_clusterModel.predict(np.array(zip(repetition_time, np.zeros(len(repetition_time)))))

            echo_time = [self.filename_pairs[index][2]["EchoTime"]]
            self.filename_pairs[index][2]["EchoTime"] = echoTime_clusterModel.predict(np.array(zip(echo_time, np.zeros(len(echo_time)))))

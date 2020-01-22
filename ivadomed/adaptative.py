import copy

import h5py
import numpy as np
import pandas as pd
from bids_neuropoly import bids
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import json

from medicaltorch import datasets as mt_datasets


class Dataframe():
    """
    This class aims to create a dataset using the bids, which can be used by an adapative loader to perform curriculum
    learning, Active Learning or any other strategy that needs to load samples in a specific way.
    It works on RAM or on the fly and can be saved for later.
    """

    def __init__(self, bids, contrasts, target_suffix, roi_suffix, path=False, ram=False):
        """
        Initialize the Dataframe
        """
        # Ram status
        self.ram = ram
        self.status = {c: self.ram for c in contrasts}
        self.status['gt'] = self.ram
        self.status['ROI'] = self.ram

        self.contrasts = contrasts
        self.df = None
        # Dataframe
        if path:
            self.load(path)
        else:
            self.create_df(bids, target_suffix, roi_suffix, ram)

    def load_column(self, column_name):
        """
        To load a column in memory
        """
        if not self.status[column_name]:
            print("TODO")
        else:
            print("Column already in RAM")

    def load_all(self):
        print("TODO")

    def shuffe(self):
        "Shuffle the whole dataframe"
        self.df = self.df.sample(frac=1)

    def load(self, path):
        """
        Load the dataframe from a csv file.
        """
        try:
            self.df = pd.read_csv(path + 'Bids_dataframe.csv')
            print("Dataframe has been correctly loaded from {}/Bids_dataframe.csv.".format(path))
        except FileNotFoundError:
            print("No csv file found")

    def save(self, path):
        """
        Save the dataframe into a csv file.
        """
        try:
            self.df.to_csv(path + '/Bids_dataframe.csv', index=True)
            print("Dataframe has been saved at {}/Bids_dataframe.csv.".format(path))
        except FileNotFoundError:
            print("Wrong path.")

    def create_df(self, bids, target_suffix, roi_suffix, ram):
        """
        Generate the Dataframe using the Bids
        """
        # Template of a line
        empty_line = {'T1w': None,
                      'T2w': None,
                      'T2star': None,
                      'gt': None,
                      'ROI': None,
                      'Slices': None,
                      'Difficulty': None}

        # Initialize the  dataframe
        col_names = ['Subjects', 'T1w', 'T2w', 'T2star', 'gt', 'ROI', 'Slices', 'Difficulty']
        df = pd.DataFrame(columns=col_names).set_index('Subjects')

        # Filling the dataframe
        for subject in bids.get_subjects():
            if subject.record["modality"] in self.contrasts:
                subject_id = subject.get_participant_id()
                line = df.loc[df.index == subject_id]

                if not line.empty:
                    df.loc[df.index == subject_id, subject.record["modality"]] = subject.record["absolute_path"]

                else:
                    if subject.has_derivative("labels"):
                        line = copy.deepcopy(empty_line)
                        line[subject.record["modality"]] = subject.record["absolute_path"]
                        derivatives = subject.get_derivatives("labels")
                        for deriv in derivatives:
                            if deriv.endswith(subject.record["modality"] + target_suffix + ".nii.gz"):
                                line['gt'] = deriv
                            if not (roi_suffix is None) and deriv.endswith(
                                    subject.record["modality"] + roi_suffix + ".nii.gz"):
                                line['ROI'] = deriv
                        df.loc[subject_id] = line

        self.df = df


class Bids_to_hdf5(Dataset):
    """
    """

    def __init__(self, root_dir, subject_lst, target_suffix, contrast_lst, hdf5_name, contrast_balance={},
                 slice_axis=2, metadata_choice=False, slice_filter_fn=None, canonical=True,
                 roi_suffix=None):

        # Getting all patients id
        self.bids_ds = bids.BIDS(root_dir)
        bids_subjects = [s for s in self.bids_ds.get_subjects() if s.record["subject_id"] in subject_lst]

        self.canonical = canonical

        # opening an hdf5 file with write access and writing metadata
        self.hdf5_file = h5py.File(hdf5_name, "w")
        self.hdf5_file.attrs['canonical'] = canonical
        list_patients = []

        self.filename_pairs = []

        if metadata_choice == 'mri_params':
            self.metadata = {"FlipAngle": [], "RepetitionTime": [],
                             "EchoTime": [], "Manufacturer": []}

        # Create a list with the filenames for all contrasts and subjects
        subjects_tot = []
        for subject in bids_subjects:
            subjects_tot.append(str(subject.record["absolute_path"]))

        # Create a dictionary with the number of subjects for each contrast of contrast_balance
        tot = {contrast: len([s for s in bids_subjects if s.record["modality"] == contrast])
               for contrast in contrast_balance.keys()}

        # Create a counter that helps to balance the contrasts
        c = {contrast: 0 for contrast in contrast_balance.keys()}

        for subject in tqdm(bids_subjects, desc="Loading dataset"):
            if subject.record["modality"] in contrast_lst:

                # Training & Validation: do not consider the contrasts over the threshold contained in contrast_balance
                if subject.record["modality"] in contrast_balance.keys():
                    c[subject.record["modality"]] = c[subject.record["modality"]] + 1
                    if c[subject.record["modality"]] / tot[subject.record["modality"]] \
                            > contrast_balance[subject.record["modality"]]:
                        continue

                if not subject.has_derivative("labels"):
                    print("Subject without derivative, skipping.")
                    continue
                derivatives = subject.get_derivatives("labels")
                target_filename, roi_filename = None, None

                for deriv in derivatives:
                    if deriv.endswith(subject.record["modality"] + target_suffix + ".nii.gz"):
                        target_filename = deriv
                    if not (roi_suffix is None) and deriv.endswith(
                            subject.record["modality"] + roi_suffix + ".nii.gz"):
                        roi_filename = deriv

                if (target_filename is None) or (not (roi_suffix is None) and (roi_filename is None)):
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

                self.filename_pairs.append((subject.record["subject_id"], [subject.record.absolute_path],
                                            target_filename, roi_filename, [metadata]))

                list_patients.append(subject.record["subject_id"])

        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.n_contrasts = len(self.filename_pairs[0][0])

        # Update HDF5 metadata
        self.hdf5_file.attrs['patients_id'] = list(set(list_patients))
        self.hdf5_file.attrs['slice_axis'] = slice_axis

        print(slice_filter_fn)
        print(type(slice_filter_fn))

        self.hdf5_file.attrs['slice_filter_fn'] = [('filter_empty_input', True), ('filter_empty_mask', False)]
        self.hdf5_file.attrs['metadata_choice'] = metadata_choice

        # Save images into HDF5 file
        self._load_filenames()
        print("files loaded")
        

    def _load_filenames(self):
        for subject_id, input_filename, gt_filename, roi_filename, metadata in self.filename_pairs:
            # Creating/ getting the subject group
            if str(subject_id) in self.hdf5_file.keys():
                grp = self.hdf5_file[str(subject_id)]
            else:
                grp = self.hdf5_file.create_group(str(subject_id))
            
            roi_pair = mt_datasets.SegmentationPair2D(input_filename, roi_filename, metadata=metadata, cache=False, canonical=self.canonical)

            seg_pair = mt_datasets.SegmentationPair2D(input_filename, gt_filename, metadata=metadata, cache=False,
                                                      canonical=self.canonical)

            input_data_shape, _ = seg_pair.get_pair_shapes()

            # TODO: adapt filter to save slices number in metadata
            useful_slices = []
            input_volumes = []
            gt_volume = []
            roi_volume = []

            for idx_pair_slice in range(input_data_shape[self.slice_axis]):
                slice_seg_pair = seg_pair.get_pair_slice(idx_pair_slice,
                                                         self.slice_axis)

                # keeping idx of slices with gt
                if self.slice_filter_fn:
                    filter_fn_ret_seg = self.slice_filter_fn(slice_seg_pair)
                if self.slice_filter_fn and not filter_fn_ret_seg:
                    useful_slices += idx_pair_slice
                    continue

                roi_pair_slice = roi_pair.get_pair_slice(idx_pair_slice, self.slice_axis)

                input_volumes.append(slice_seg_pair["input"][0])

                # Handle unlabeled data
                if slice_seg_pair["gt"] is None:
                    gt_img = None
                else:
                    gt_volume.append((slice_seg_pair["gt"] * 255).astype(np.uint8))

                # Handle data with no ROI provided
                if roi_pair_slice["gt"] is None:
                    roi_img = None
                else:
                    roi_volume.append((roi_pair_slice["gt"] * 255).astype(np.uint8))

            # Getting metadata using the one from the last slice
            input_metadata = slice_seg_pair['input_metadata']
            gt_metadata = slice_seg_pair['gt_metadata']
            roi_metadata = roi_pair_slice['input_metadata'][0]

            if grp.attrs.__contains__('slices'):
                grp.attrs['slices'] = list(set(grp.attrs['slices'] + useful_slices))
            else:
                grp.attrs['slices'] = useful_slices

            # Creating datasets and metadata
            # Inputs
            
            print(input_metadata[0]['contrast'])
            key = "inputs/{}".format(input_metadata[0]['contrast'])
            grp.create_dataset(key, data=input_volumes)
            # Sub-group metadata
            if grp['inputs'].attrs.__contains__('contrast'):
                grp['inputs'].attrs['contrast'].append(input_metadata[0]['contrast'])
            else:
                grp['inputs'].attrs['contrast'] = [input_metadata[0]['contrast']]

            # dataset metadata
            grp[key].attrs['input_filename'] = input_metadata[0]['input_filename']
            ### TODO: Add other metadata

            # GT
            print(gt_metadata.keys())
            contrast = input_metadata[0]['contrast']
            key = "gt/{}".format(contrast)
            grp.create_dataset(key, data=gt_volume)
            # Sub-group metadata
            if grp['gt'].attrs.__contains__('contrast'):
                grp['gt'].attrs['contrast'].append(contrast)
            else:
                grp['gt'].attrs['contrast'] = [contrast]

            # dataset metadata
            grp[key].attrs['gt_filename'] = input_metadata[0]['gt_filename']
            ### TODO: Add other metadata

            # ROI
            key = "roi/{}".format(contrast)
            grp.create_dataset(key, data=roi_volume)
            # Sub-group metadata
            if grp['roi'].attrs.__contains__('contrast'):
                grp['roi'].attrs['contrast'].append(contrast)
            else:
                grp['roi'].attrs['contrast'] = [contrast]

            # dataset metadata
            grp[key].attrs['input_filename'] = roi_metadata['input_filename']
            ### TODO: Add other metadata


class BidsDataset(mt_datasets.MRI2DSegmentationDataset):

    def filter_roi(self, nb_nonzero_thr):
        filter_indexes = []
        for segpair, slice_roi_pair in self.indexes:
            roi_data = slice_roi_pair['gt']
            if not np.any(roi_data):
                continue
            if np.count_nonzero(roi_data) <= nb_nonzero_thr:
                continue

            filter_indexes.append((segpair, slice_roi_pair))

        self.indexes = filter_indexes


class HDF5Dataset():
    def __init__(self, dataroot, filename, RAM=True):

        if not os.path.isfile(dataroot):
            print("Computing hdf5 file of the data")
            dataset = json.load(open(self.dataroot + "dataset.json"))
            files = dataset['training']
            Bids_to_hdf5(dataroot, files, filename)
        else:
            hf = h5py.File(filename, "r")

        self.dict = hf
        if RAM:
            self.load_into_ram()
        # TODO list:
        """ 
        - implement load_into_ram() & partial mode

        - include dataframe class
            - Mod can refer to either the path of the image in the HDF5 file or 
        - transform numpy into PIL image

        return dict like 
            data_dict = {
                'input': input_tensors,
                'gt': gt_img,
                'roi': roi_img,
                'input_metadata': input_metadata,
                'gt_metadata': seg_pair_slice['gt_metadata'],
                'roi_metadata': roi_pair_slice['gt_metadata']
            }
            return data_dict

        """

    def load_into_ram(self):
        self.gt = []

    def set_transform(self, transform):
        """ This method will replace the current transformation for the
        dataset.

        :param transform: the new transformation
        """
        self.transform = transform

    def __len__(self):
        """Return the dataset size. The number of subvolumes."""
        return len(self.indexes)

    def __getitem__(self, index):
        self.hf[index]


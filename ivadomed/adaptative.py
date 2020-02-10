import copy
from os import path

import h5py
import numpy as np
import pandas as pd
from bids_neuropoly import bids
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import json
from PIL import Image

from medicaltorch import datasets as mt_datasets


class Dataframe:
    """
    This class aims to create a dataset using an HDF5 file, which can be used by an adapative loader
    to perform curriculum learning, Active Learning or any other strategy that needs to load samples in a specific way.
    It works on RAM or on the fly and can be saved for later.
    """

    def __init__(self, hdf5, contrasts, path, target_suffix=None, roi_suffix=None,
                 slices=False, dim=2):
        """
        Initialize the Dataframe
        """
        # Number of dimension
        self.dim = dim
        # List of all contrasts
        self.contrasts = contrasts

        if target_suffix:
            for gt in target_suffix:
                self.contrasts.append('gt/' + gt)
        else:
            self.contrasts.append('gt')

        if roi_suffix:
            for roi in roi_suffix:
                self.contrasts.append('roi/' + roi)
        else:
            self.contrasts.append('ROI')

        self.df = None
        self.filter = slices

        # Data frame
        if os.path.exists(path):
            self.load_dataframe(path)
        else:
            self.create_df(hdf5)

    def shuffe(self):
        """Shuffle the whole data frame"""
        self.df = self.df.sample(frac=1)

    def load_dataframe(self, path):
        """
        Load the dataframe from a csv file.
        """
        try:
            self.df = pd.read_csv(path)
            print("Dataframe has been correctly loaded from {}.".format(path))
        except FileNotFoundError:
            print("No csv file found")

    def save(self, path):
        """
        Save the dataframe into a csv file.
        """
        try:
            self.df.to_csv(path, index=False)
            print("Dataframe has been saved at {}.".format(path))
        except FileNotFoundError:
            print("Wrong path.")

    def create_df(self, hdf5):
        """
        Generate the Data frame using the hdf5 file
        """
        # Template of a line
        empty_line = {col: None for col in self.contrasts}
        empty_line['Slices'] = None

        # Initialize the data frame
        col_names = [col for col in empty_line.keys()]
        col_names.append('Subjects')
        df = pd.DataFrame(columns=col_names)

        # Filling the data frame
        for subject in hdf5.attrs['patients_id']:
            # Getting the Group the corresponding patient
            grp = hdf5[subject]
            line = copy.deepcopy(empty_line)
            line['Subjects'] = subject

            # inputs
            assert 'inputs' in grp.keys()
            inputs = grp['inputs']
            for c in inputs.attrs['contrast']:
                if c in col_names:
                    line[c] = '{}/inputs/{}'.format(subject, c)
                else:
                    continue
            # GT
            assert 'gt' in grp.keys()
            inputs = grp['gt']
            for c in inputs.attrs['contrast']:
                key = 'gt/' + c
                if key in col_names:
                    line[key] = '{}/gt/{}'.format(subject, c)
                else:
                    continue
            # ROI
            assert 'roi' in grp.keys()
            inputs = grp['roi']
            for c in inputs.attrs['contrast']:
                key = 'roi/' + c
                if key in col_names:
                    line[key] = '{}/roi/{}'.format(subject, c)
                else:
                    continue

            # Adding slices & removing useless slices if loading in ram
            line['Slices'] = np.array(grp.attrs['slices'])

            # If the number of dimension is 2, we separate the slices
            if self.dim == 2:
                if self.filter:
                    for n in line['Slices']:
                        line_slice = copy.deepcopy(line)
                        line_slice['Slices'] = n
                        df = df.append(line_slice, ignore_index=True)

            else:
                df = df.append(line, ignore_index=True)

        self.df = df


class Bids_to_hdf5:
    """

    """

    def __init__(self, root_dir, subject_lst, target_suffix, contrast_lst, hdf5_name, contrast_balance={},
                 slice_axis=2, metadata_choice=False, slice_filter_fn=None, canonical=True,
                 roi_suffix=None):
        """

        :param root_dir: path of the bids
        :param subject_lst: list of patients
        :param target_suffix: suffix of the gt
        :param roi_suffix: suffix of the roi
        :param contrast_lst: list of the contrast
        :param hdf5_name: path and name of the hdf5 file
        :param contrast_balance:
        :param slice_axis:
        :param metadata_choice:
        :param slice_filter_fn:
        :param canonical:

        """

        print("Starting conversion")
        # Getting all patients id
        self.bids_ds = bids.BIDS(root_dir)
        bids_subjects = [s for s in self.bids_ds.get_subjects() if s.record["subject_id"] in subject_lst]

        self.canonical = canonical
        self.dt = h5py.special_dtype(vlen=str)
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

        # Update HDF5 metadata
        self.hdf5_file.attrs.create('patients_id', list(set(list_patients)), dtype=self.dt)
        self.hdf5_file.attrs['slice_axis'] = slice_axis

        self.hdf5_file.attrs['slice_filter_fn'] = [('filter_empty_input', True), ('filter_empty_mask', False)]
        self.hdf5_file.attrs['metadata_choice'] = metadata_choice

        # Save images into HDF5 file
        self._load_filenames()
        print("Files loaded.")

    def _load_filenames(self):
        for subject_id, input_filename, gt_filename, roi_filename, metadata in self.filename_pairs:
            # Creating/ getting the subject group
            if str(subject_id) in self.hdf5_file.keys():
                grp = self.hdf5_file[str(subject_id)]
            else:
                grp = self.hdf5_file.create_group(str(subject_id))

            roi_pair = mt_datasets.SegmentationPair2D(input_filename, roi_filename, metadata=metadata, cache=False,
                                                      canonical=self.canonical)

            seg_pair = mt_datasets.SegmentationPair2D(input_filename, gt_filename, metadata=metadata, cache=False,
                                                      canonical=self.canonical)

            input_data_shape, _ = seg_pair.get_pair_shapes()

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
                if self.slice_filter_fn and filter_fn_ret_seg:
                    useful_slices.append(idx_pair_slice)

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
            input_metadata = slice_seg_pair['input_metadata'][0]
            gt_metadata = slice_seg_pair['gt_metadata']
            roi_metadata = roi_pair_slice['input_metadata'][0]

            if grp.attrs.__contains__('slices'):
                grp.attrs['slices'] = list(set(np.concatenate((grp.attrs['slices'], useful_slices))))
            else:
                grp.attrs['slices'] = useful_slices

            # Creating datasets and metadata
            contrast = input_metadata['contrast']
            # Inputs
            key = "inputs/{}".format(contrast)
            grp.create_dataset(key, data=input_volumes)
            # Sub-group metadata
            if grp['inputs'].attrs.__contains__('contrast'):
                attr = grp['inputs'].attrs['contrast']
                new_attr = [c for c in attr]
                new_attr.append(contrast)
                grp['inputs'].attrs.create('contrast', new_attr, dtype=self.dt)

            else:
                grp['inputs'].attrs.create('contrast', [contrast], dtype=self.dt)

            # dataset metadata
            grp[key].attrs['input_filename'] = input_metadata['input_filename']

            if "zooms" in input_metadata.keys():
                grp[key].attrs["zooms"] = input_metadata['zooms']
            if "data_shape" in input_metadata.keys():
                grp[key].attrs["data_shape"] = input_metadata['data_shape']

            # GT
            key = "gt/{}".format(contrast)
            grp.create_dataset(key, data=gt_volume)
            # Sub-group metadata
            if grp['gt'].attrs.__contains__('contrast'):
                attr = grp['gt'].attrs['contrast']
                new_attr = [c for c in attr]
                new_attr.append(contrast)
                grp['gt'].attrs.create('contrast', new_attr, dtype=self.dt)

            else:
                grp['gt'].attrs.create('contrast', [contrast], dtype=self.dt)

            # dataset metadata
            grp[key].attrs['gt_filename'] = input_metadata['gt_filename']

            if "zooms" in gt_metadata.keys():
                grp[key].attrs["zooms"] = gt_metadata['zooms']
            if "data_shape" in gt_metadata.keys():
                grp[key].attrs["data_shape"] = gt_metadata['data_shape']

            # ROI
            key = "roi/{}".format(contrast)
            grp.create_dataset(key, data=roi_volume)
            # Sub-group metadata
            if grp['roi'].attrs.__contains__('contrast'):
                attr = grp['roi'].attrs['contrast']
                new_attr = [c for c in attr]
                new_attr.append(contrast)
                grp['roi'].attrs.create('contrast', new_attr, dtype=self.dt)

            else:
                grp['roi'].attrs.create('contrast', [contrast], dtype=self.dt)

            # dataset metadata
            grp[key].attrs['roi_filename'] = roi_metadata['gt_filename']

            if "zooms" in roi_metadata.keys():
                grp[key].attrs["zooms"] = roi_metadata['zooms']
            if "data_shape" in roi_metadata.keys():
                grp[key].attrs["data_shape"] = roi_metadata['data_shape']

            # Adding contrast to group metadata
            if grp.attrs.__contains__('contrast'):
                attr = grp.attrs['contrast']
                new_attr = [c for c in attr]
                new_attr.append(contrast)
                grp.attrs.create('contrast', new_attr, dtype=self.dt)

            else:
                grp.attrs.create('contrast', [contrast], dtype=self.dt)


class HDF5Dataset:
    def __init__(self, root_dir, subject_lst, hdf5_name, csv_name, target_suffix, contrast_lst, ram=True,
                 contrast_balance={}, slice_axis=2, transform=None, metadata_choice=False, dim=2,
                 slice_filter_fn=None, canonical=True, roi_suffix=None, target_lst=None, roi_lst=None):

        """

        :param root_dir: path of bids and data
        :param subject_lst: list of patients
        :param hdf5_name: path of the hdf5 file
        :param csv_name: path of the dataframe
        :param target_suffix: suffix of the gt
        :param roi_suffix: suffix of the roi
        :param contrast_lst: list of the contrast
        :param contrast_balance: contrast balance
        :param slice_axis: slice axis. by default it's set to 2
        :param transform: transformation
        :param dim: number of dimension of our data. Either 2 or 3
        :param metadata_choice:
        :param slice_filter_fn:
        :param canonical:
        :param roi_suffix:
        :param target_lst:
        :param roi_lst:
        """

        self.cst_lst = copy.deepcopy(contrast_lst)
        self.gt_lst = copy.deepcopy(target_lst)
        self.roi_lst = copy.deepcopy(roi_lst)

        self.dim = dim
        self.filter_slices = slice_filter_fn
        self.transform = transform
        # Getting HDS5 dataset file
        if not os.path.isfile(hdf5_name):
            print("Computing hdf5 file of the data")
            hdf5_file = Bids_to_hdf5(root_dir,
                                          subject_lst=subject_lst,
                                          hdf5_name=hdf5_name,
                                          target_suffix=target_suffix,
                                          roi_suffix=roi_suffix,
                                          contrast_lst=contrast_lst,
                                          metadata_choice=metadata_choice,
                                          contrast_balance=contrast_balance,
                                          slice_axis=slice_axis,
                                          canonical=canonical,
                                          slice_filter_fn=slice_filter_fn
                                          )
            self.hdf5_file = hdf5_file.hdf5_file
        else:
            self.hdf5_file = h5py.File(hdf5_name, "r")
        # Loading dataframe object
        self.df_object = Dataframe(self.hdf5_file, contrast_lst, csv_name, target_suffix=target_lst,
                                   roi_suffix=roi_lst, dim=self.dim, slices=slice_filter_fn)
        self.initial_dataframe = self.df_object.df
        self.dataframe = copy.deepcopy(self.df_object.df)

        self.cst_matrix = np.ones([len(self.dataframe), len(self.cst_lst)], dtype=int)

        # RAM status
        self.status = {ct: False for ct in self.df_object.contrasts}
        # Input contrasts
        self.contrast_lst = contrast_lst

    def load_into_ram(self, contrast_lst=None):
        """
        Aims to load into RAM the contrasts from the list
        :param contrast_lst: list of contrast
        :return:
        """
        keys = self.status.keys()
        for ct in contrast_lst:
            if ct not in keys:
                print("Key error: status has no key {}".format(ct))
                continue
            if self.status[ct]:
                print("Contrast {} already in RAM".format(ct))
            else:
                print("Loading contrast {} in RAM...".format(ct), end='')
                for sub in self.dataframe.index:
                    if self.filter_slices:
                        slices = self.dataframe.at[sub, 'Slices']
                        self.dataframe.at[sub, ct] = self.hdf5_file[self.dataframe.at[sub, ct]][np.array(slices)]
                print("Done.")
            self.status[ct] = True

    def set_transform(self, transform):
        """ This method will replace the current transformation for the
        dataset.

        :param transform: the new transformation
        """
        self.transform = transform

    def __len__(self):
        """Return the dataset size. The number of subvolumes."""
        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Warning: For now, this method only supports one gt/ roi

        :param index:
        :return: data_dict = {'input': input_tensors,
                                'gt': gt_img,
                                'roi': roi_img,
                                'input_metadata': input_metadata,
                                'gt_metadata': seg_pair_slice['gt_metadata'],
                                'roi_metadata': roi_pair_slice['gt_metadata']
                                }
        """
        line = self.dataframe.loc[index]
        # For HeMIS strategy. Otherwise the values of the matrix dont change anything.
        missing_modalities = self.cst_matrix[index]

        input_metadata = []
        input_tensors = []

        # Inputs
        for i, ct in enumerate(self.cst_lst):
            if self.status[ct]:
                input_tensor = line[ct] * missing_modalities[i]
            else:
                input_tensor = self.hdf5_file[line[ct]][line['Slices']] * missing_modalities[i]

            # convert array to pil
            input_tensors.append(Image.fromarray(input_tensor, mode='F'))
            # input Metadata
            input_metadata.append({key: value for key, value in self.hdf5_file['{}/inputs/{}'
                                  .format(line['Subjects'], ct)].attrs.items()})

        # GT
        if self.status['gt/' + self.gt_lst[0]]:
            gt_img = line['gt/' + self.gt_lst[0]]
        else:
            gt_img = self.hdf5_file[line['gt/' + self.gt_lst[0]]][line['Slices']]

        # convert array to pil
        gt_img = (gt_img * 255).astype(np.uint8)
        gt_img = Image.fromarray(gt_img, mode='L')
        gt_metadata = {key: value for key, value in
                       self.hdf5_file['{}/gt/{}'.format(line['Subjects'], self.gt_lst[0])].attrs.items()}

        # ROI
        if self.status['roi/' + self.roi_lst[0]]:
            roi_img = line['roi/' + self.roi_lst[0]]
        else:
            roi_img = self.hdf5_file[line['roi/' + self.roi_lst[0]]][line['Slices']]

        # convert array to pil
        roi_img = (roi_img * 255).astype(np.uint8)
        roi_img = Image.fromarray(roi_img, mode='L')

        roi_metadata = {key: value for key, value in
                        self.hdf5_file['{}/roi/{}'.format(line['Subjects'], self.roi_lst[0])].attrs.items()}

        data_dict = {'input': input_tensors,
                     'gt': gt_img,
                     'roi': roi_img,
                     'input_metadata': input_metadata,
                     'Missing_mod': missing_modalities,
                     'gt_metadata': gt_metadata,
                     'roi_metadata': roi_metadata
                     }

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict

    def update(self, strategy="Missing", p=0.0001):
        """
        Update the Dataframe itself.
        :param p: probability of the modality to be missing
        :param strategy: Update the dataframe using the corresponding strategy. For now the only the strategy
        implemented is the one used by HeMIS (i.e. by removing modalities with a certain probability.) Other strategies
        that could be implemented are Active Learning, Curriculum Learning, ...

        """
        if strategy == 'Missing':
            print("Probalility of missing modality = {}".format(p))
            self.cst_matrix = np.array([np.random.choice(2, len(self.cst_lst), p=[p, 1 - p])
                                        for _ in range(len(self.dataframe))])
            print("Missing modalities = {}".format(self.cst_matrix.size - self.cst_matrix.sum()))


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


def HDF5_to_Bids(HDF5, subjects, path_dir):
    # Open FDH5 file
    hdf5 = h5py.File(HDF5, "r")
    # check the dir exists:
    if not path.exists(path_dir):
        print("Directory doesn't exist. Stopping process.")
        exit(0)
    # loop over all subjects
    for sub in subjects:
        path_sub = path_dir + '/' + sub
        path_label = path_dir + '/derivatives/labels/' + sub + '/anat/'

        if not path.exists(path_sub):
            os.mkdir(path_sub)

        if not path.exists(path_label):
            os.mkdir(path_label)

        # Get Subject Group
        grp = hdf5['sub']
        # inputs
        cts = grp['inputs'].attrs['contrast']

        for ct in cts:
            input = grp['inputs/{}'.format(ct)]

            ## Save nii with save_nii

        # GT
        cts = grp['gt'].attrs['contrast']

        for ct in cts:
            gt = grp['gt/{}'.format(ct)]

            ## Save nii with save_nii

        cts = grp['roi'].attrs['contrast']

        for ct in cts:
            input = grp['roi/{}'.format(ct)]

            ## Save nii with save_nii

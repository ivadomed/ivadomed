import numpy as np
import pandas as pd
import json
import copy

from glob import glob
from copy import deepcopy

from medicaltorch.datasets import SampleMetadata
from tqdm import tqdm
import nibabel as nib
import torch

from torch.utils.data import Dataset
import os, os.path
import skimage, skimage.transform
from skimage.io import imread, imsave
from PIL import Image
import skimage.filters
import json

import collections
import h5py, ntpath

from bids_neuropoly import bids
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


    def __init__(self, root_dir, subject_lst, target_suffix, contrast_lst, hdf5_name, contrast_balance={}, slice_axis=2,
                 cache=True, metadata_choice=False, slice_filter_fn=None,
                 canonical=True, roi_suffix=None, multichannel=False):

        self.canonical = canonical
        self.bids_ds = bids.BIDS(root_dir)
        bids_subjects = [s for s in self.bids_ds.get_subjects() if s.record["subject_id"] in subject_lst]

        # opening an hdf5 file with write access and writing metadata
        self.hdf5_file = h5py.File(hdf5_name, "w")
        self.hdf5_file.attrs['canonical'] = canonical
        self.hdf5_file.attrs['patients_id'] = bids_subjects

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
                    self.filename_pairs.append((subject.record["subject_id"], [subject.record.absolute_path],
                                                target_filename, roi_filename, [metadata]))

        if multichannel:
            for subject in multichannel_subjects.values():
                if not None in subject["absolute_paths"]:
                    self.filename_pairs.append((subject.record["subject_id"], subject["absolute_paths"], subject["deriv_path"],
                                                subject["roi_filename"], subject["metadata"]))

        self.indexes = []
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.n_contrasts = len(self.filename_pairs[0][0])

        self._load_filenames()

    def _load_filenames(self):
        for subject_id, input_filename, gt_filename, roi_filename, metadata in self.filename_pairs:
            # Creating a new group for the subject
            try:
                grp = self.hdf5_file.create_group(str(subject_id))
            except:
                grp = self.hdf5_file[str(subject_id)]
                print("Group already exist - Loading it.")

            roi_pair = SegmentationPair2D(input_filename, roi_filename, canonical=self.canonical)

            seg_pair = SegmentationPair2D(input_filename, gt_filename, canonical=self.canonical)

            input_data_shape, _ = seg_pair.get_pair_shapes()


            # TODO: adapt filter to save slices number in metadata
            useful_slices = []
            input_volumes = [[] for _ in range(len(seg_pair["input"]))]
            gt_volume = []
            roi_volume = []
            input_metadata = None

            for idx_pair_slice in range(input_data_shape[self.slice_axis]):
                slice_seg_pair = seg_pair.get_pair_slice(idx_pair_slice,
                                                         self.slice_axis)
                if self.slice_filter_fn:
                    filter_fn_ret_seg = self.slice_filter_fn(slice_seg_pair)
                if self.slice_filter_fn and not filter_fn_ret_seg:
                    useful_slices += idx_pair_slice
                    continue

                roi_pair_slice = roi_pair.get_pair_slice(idx_pair_slice, self.slice_axis)




                # Looping over all modalities (one or more)
                for idx, input_slice in enumerate(slice_seg_pair["input"]):
                    input_img = Image.fromarray(input_slice, mode='F')
                    input_volumes[idx].append(input_img)


                # Handle unlabeled data
                if slice_seg_pair["gt"] is None:
                    gt_img = None

                else:
                    gt_volume.append((slice_seg_pair["gt"] * 255).astype(np.uint8))
                    #gt_img = Image.fromarray(gt_img, mode='L')

                # Handle data with no ROI provided
                if roi_pair_slice["gt"] is None:
                    roi_img = None
                else:
                    roi_volume.append((roi_pair_slice["gt"] * 255).astype(np.uint8))
                    #roi_img = Image.fromarray(roi_img, mode='L')

            grp.attrs['slices'] = useful_slices
            # Creating datasets and metadata
            # Inputs
            for i in range(len(input_volumes)):
                key = "inputs/mydataset"
                grp.create_dataset(key, data=input_volumes)
                grp[key].attrs['']
            # GT
            grp.create_dataset("gt/mydataset", data=gt_volume)

            # ROI
            grp.create_dataset("roi/mydataset", data=roi_volume)


            # TODO: Move that part into HDF5Dataset
            """
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



class SegmentationPair2D(object):
    """This class is used to build 2D segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).

    :param input_filenames: the input filename list (supported by nibabel). For single channel, the list will contain 1
                           input filename.
    :param gt_filename: the ground-truth filename.
    :param metadata: metadata list with each item corresponding to an image (modality) in input_filenames.  For single channel, the list will contain metadata related to
                     to one image.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """

    def __init__(self, input_filenames, gt_filename, cache=False, canonical=False):

        self.input_filenames = input_filenames
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache

        # list of the images
        self.input_handle = []

        # loop over the filenames (list)
        for input_file in self.input_filenames:
            input_img = nib.load(input_file)
            self.input_handle.append(input_img)
            if len(input_img.shape) > 3:
                raise RuntimeError("4-dimensional volumes not supported.")

        # we consider only one gt per patient
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


        """
        To remove
        if self.metadata:
            self.metadata = []
            for data,input_filename in zip(metadata,input_filenames):
                data["input_filename"] = input_filename
                data["gt_filename"] = gt_filename
                self.metadata.append(data)
        """
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
        """Return the tuple (input, ground truth) with the data content in
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
            input_dataobj = [handle.dataobj for handle in self.input_handle]

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        input_slice = []
        # Loop over modalities
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

            gt_meta_dict = SampleMetadata({
                "zooms": self.gt_handle.header.get_zooms()[:2],
                "data_shape": self.gt_handle.header.get_data_shape()[:2],
            })

        input_meta_dict = []
        for handle in self.input_handle:
            input_meta_dict.append(SampleMetadata({
                "zooms": handle.header.get_zooms()[:2],
                "data_shape": handle.header.get_data_shape()[:2],
            }))

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }

        if self.metadata:
            for idx, metadata in enumerate(self.metadata):  # loop across channels
                metadata["slice_index"] = slice_index
                self.metadata[idx] = metadata
                for metadata_key in metadata.keys():  # loop across input metadata
                    dreturn["input_metadata"][idx][metadata_key] = metadata[metadata_key]

        return dreturn



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


class HDF5Dataset(mt_datasets):
    def __init__(dataroot, RAM=True):

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
        #TODO list:
        """ 
        - implement load_into_ram() & partial mode
        - include dataframe class
            - Mod can refer to either the path of the image in the HDF5 file or 
        - transform numpy into PIL image
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


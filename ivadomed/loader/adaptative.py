import copy
import os
from os import path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from bids_neuropoly import bids
from tqdm import tqdm

from ivadomed import transforms as imed_transforms
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader, film as imed_film
from ivadomed.object_detection import utils as imed_obj_detect


class Dataframe:
    """
    This class aims to create a dataset using an HDF5 file, which can be used by an adapative loader
    to perform curriculum learning, Active Learning or any other strategy that needs to load samples in a specific way.
    It works on RAM or on the fly and can be saved for later.

    Args:
        hdf5 (hdf5): hdf5 file containing dataset information
        contrasts (list of str): List of the contrasts of interest.
        path (str): Dataframe path.
        target_suffix (list of str): List of suffix of targetted structures.
        roi_suffix (str): List of suffix of ROI masks.
        filter_slices (SliceFilter): Object that filters slices according to their content.
        dim (int): Choice 2 or 3, for 2D or 3D data respectively.

    Attributes:
        dim (int): Choice 2 or 3, for 2D or 3D data respectively.
        contrasts (list of str): List of the contrasts of interest.
        filter_slices (SliceFilter): Object that filters slices according to their content.
        df (pd.Dataframe): Dataframe containing dataset information
    """

    def __init__(self, hdf5, contrasts, path, target_suffix=None, roi_suffix=None,
                 filter_slices=False, dim=2):
        # Number of dimension
        self.dim = dim
        # List of all contrasts
        self.contrasts = copy.deepcopy(contrasts)

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
        self.filter = filter_slices

        # Data frame
        if os.path.exists(path):
            self.load_dataframe(path)
        else:
            self.create_df(hdf5)

    def shuffle(self):
        """Shuffle the whole data frame."""
        self.df = self.df.sample(frac=1)

    def load_dataframe(self, path):
        """Load the dataframe from a csv file.

        Args:
            path (str): Path to hdf5 file.
        """
        try:
            self.df = pd.read_csv(path)
            print("Dataframe has been correctly loaded from {}.".format(path))
        except FileNotFoundError:
            print("No csv file found")

    def save(self, path):
        """Save the dataframe into a csv file.

        Args:
            path (str): Path to hdf5 file.
        """
        try:
            self.df.to_csv(path, index=False)
            print("Dataframe has been saved at {}.".format(path))
        except FileNotFoundError:
            print("Wrong path.")

    def create_df(self, hdf5):
        """Generate the Data frame using the hdf5 file.

        Args:
            hdf5 (hdf5): File containing dataset information
        """
        # Template of a line
        empty_line = {col: 'None' for col in self.contrasts}
        empty_line['Slices'] = 'None'

        # Initialize the data frame
        col_names = [col for col in empty_line.keys()]
        col_names.append('Subjects')
        df = pd.DataFrame(columns=col_names)
        print(hdf5.attrs['patients_id'])
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
                for col in col_names:
                    if key in col:
                        line[col] = '{}/gt/{}'.format(subject, c)
                    else:
                        continue
            # ROI
            assert 'roi' in grp.keys()
            inputs = grp['roi']
            for c in inputs.attrs['contrast']:
                key = 'roi/' + c
                for col in col_names:
                    if key in col:
                        line[col] = '{}/roi/{}'.format(subject, c)
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

    def clean(self, contrasts):
        """Aims to remove lines where one of the contrasts in not available.

        Agrs:
            contrasts (list of str): List of contrasts.
        """
        # Replacing 'None' values by np.nan
        self.df[contrasts] = self.df[contrasts].replace(to_replace='None', value=np.nan)
        # Dropping np.nan
        self.df = self.df.dropna()


class Bids_to_hdf5:
    """Converts a BIDS dataset to a HDF5 file.

    Args:
        root_dir (str): Path to the BIDS dataset.
        subject_lst (list): Subject names list.
        target_suffix (list): List of suffixes for target masks.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.
        contrast_lst (list): List of the contrasts.
        hdf5_name (str): Path and name of the hdf5 file.
        contrast_balance (dict): Dictionary controlling image contrasts balance.
        slice_axis (int): Indicates the axis used to extract slices: "axial": 2, "sagittal": 0, "coronal": 1.
        metadata_choice (str): Choice between "mri_params", "contrasts", None or False, related to FiLM.
        slice_filter_fn (SliceFilter): Class that filters slices according to their content.
        transform (Compose): Transformations.
        object_detection_params (dict): Object detection parameters.

    Attributes:
        bids_ds (BIDS): BIDS dataset.
        dt (dtype): hdf5 special dtype.
        hdf5_file (hdf5): hdf5 file containing dataset information.
        filename_pairs (list): A list of tuples in the format (input filename list containing all modalities,ground \
            truth filename, ROI filename, metadata).
        metadata (dict): Dictionary containing metadata of input and gt.
        prepro_transforms (Compose): Transforms to be applied before training.
        transform (Compose): Transforms to be applied during training.
        has_bounding_box (bool): True if all metadata contains bounding box coordinates, else False.
        slice_axis (int): Indicates the axis used to extract slices: "axial": 2, "sagittal": 0, "coronal": 1.
        slice_filter_fn (SliceFilter): Object that filters slices according to their content.
    """

    def __init__(self, root_dir, subject_lst, target_suffix, contrast_lst, hdf5_name, contrast_balance=None,
                 slice_axis=2, metadata_choice=False, slice_filter_fn=None, roi_params=None, transform=None,
                 object_detection_params=None, soft_gt=False):
        print("Starting conversion")
        # Getting all patients id
        self.bids_ds = bids.BIDS(root_dir)
        bids_subjects = [s for s in self.bids_ds.get_subjects() if s.record["subject_id"] in subject_lst]
        self.soft_gt = soft_gt
        self.dt = h5py.special_dtype(vlen=str)
        # opening an hdf5 file with write access and writing metadata
        self.hdf5_file = h5py.File(hdf5_name, "w")

        list_patients = []

        self.filename_pairs = []

        if metadata_choice == 'mri_params':
            self.metadata = {"FlipAngle": [], "RepetitionTime": [],
                             "EchoTime": [], "Manufacturer": []}

        self.prepro_transforms, self.transform = transform
        # Create a list with the filenames for all contrasts and subjects
        subjects_tot = []
        for subject in bids_subjects:
            subjects_tot.append(str(subject.record["absolute_path"]))

        # Create a dictionary with the number of subjects for each contrast of contrast_balance
        tot = {contrast: len([s for s in bids_subjects if s.record["modality"] == contrast])
               for contrast in contrast_balance.keys()}

        # Create a counter that helps to balance the contrasts
        c = {contrast: 0 for contrast in contrast_balance.keys()}

        self.has_bounding_box = True
        bounding_box_dict = imed_obj_detect.load_bounding_boxes(object_detection_params,
                                                                self.bids_ds.get_subjects(),
                                                                slice_axis,
                                                                contrast_lst)

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

                target_filename, roi_filename = [None] * len(target_suffix), None

                for deriv in derivatives:
                    for idx, suffix in enumerate(target_suffix):
                        if deriv.endswith(subject.record["modality"] + suffix + ".nii.gz"):
                            target_filename[idx] = deriv

                    if not (roi_params["suffix"] is None) and \
                            deriv.endswith(subject.record["modality"] + roi_params["suffix"] + ".nii.gz"):
                        roi_filename = [deriv]

                if (not any(target_filename)) or (not (roi_params["suffix"] is None) and (roi_filename is None)):
                    continue

                if not subject.has_metadata():
                    print("Subject without metadata.")
                    metadata = {}
                else:
                    metadata = subject.metadata()
                    # add contrast to metadata
                metadata['contrast'] = subject.record["modality"]

                if metadata_choice == 'mri_params':
                    if not all([imed_film.check_isMRIparam(m, metadata) for m in self.metadata.keys()]):
                        continue

                if len(bounding_box_dict):
                    # Take only one bounding box for cropping
                    metadata['bounding_box'] = bounding_box_dict[str(subject.record["absolute_path"])][0]

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
        """Load preprocessed pair data (input and gt) in handler."""
        for subject_id, input_filename, gt_filename, roi_filename, metadata in self.filename_pairs:
            # Creating/ getting the subject group
            if str(subject_id) in self.hdf5_file.keys():
                grp = self.hdf5_file[str(subject_id)]
            else:
                grp = self.hdf5_file.create_group(str(subject_id))

            roi_pair = imed_loader.SegmentationPair(input_filename, roi_filename, metadata=metadata,
                                                    slice_axis=self.slice_axis, cache=False, soft_gt=self.soft_gt)

            seg_pair = imed_loader.SegmentationPair(input_filename, gt_filename, metadata=metadata,
                                                    slice_axis=self.slice_axis, cache=False, soft_gt=self.soft_gt)
            print("gt filename", gt_filename)
            input_data_shape, _ = seg_pair.get_pair_shapes()

            useful_slices = []
            input_volumes = []
            gt_volume = []
            roi_volume = []

            for idx_pair_slice in range(input_data_shape[-1]):

                slice_seg_pair = seg_pair.get_pair_slice(idx_pair_slice)

                self.has_bounding_box = imed_obj_detect.verify_metadata(slice_seg_pair, self.has_bounding_box)
                if self.has_bounding_box:
                    imed_obj_detect.adjust_transforms(self.prepro_transforms, slice_seg_pair)

                # keeping idx of slices with gt
                if self.slice_filter_fn:
                    filter_fn_ret_seg = self.slice_filter_fn(slice_seg_pair)
                if self.slice_filter_fn and filter_fn_ret_seg:
                    useful_slices.append(idx_pair_slice)

                roi_pair_slice = roi_pair.get_pair_slice(idx_pair_slice)
                slice_seg_pair, roi_pair_slice = imed_transforms.apply_preprocessing_transforms(self.prepro_transforms,
                                                                                                slice_seg_pair,
                                                                                                roi_pair_slice)

                input_volumes.append(slice_seg_pair["input"][0])

                # Handle unlabeled data
                if not len(slice_seg_pair["gt"]):
                    gt_volume = []
                else:
                    gt_volume.append((slice_seg_pair["gt"][0] * 255).astype(np.uint8) / 255.)

                # Handle data with no ROI provided
                if not len(roi_pair_slice["gt"]):
                    roi_volume = []
                else:
                    roi_volume.append((roi_pair_slice["gt"][0] * 255).astype(np.uint8) / 255.)

            # Getting metadata using the one from the last slice
            input_metadata = slice_seg_pair['input_metadata'][0]
            gt_metadata = slice_seg_pair['gt_metadata'][0]
            roi_metadata = roi_pair_slice['input_metadata'][0]

            if grp.attrs.__contains__('slices'):
                grp.attrs['slices'] = list(set(np.concatenate((grp.attrs['slices'], useful_slices))))
            else:
                grp.attrs['slices'] = useful_slices

            # Creating datasets and metadata
            contrast = input_metadata['contrast']
            # Inputs
            print(len(input_volumes))
            print("grp= ", str(subject_id))
            key = "inputs/{}".format(contrast)
            print("key = ", key)
            if len(input_volumes) < 1:
                print("list empty")
                continue
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
            grp[key].attrs['input_filenames'] = input_metadata['input_filenames']
            grp[key].attrs['data_type'] = input_metadata['data_type']

            if "zooms" in input_metadata.keys():
                grp[key].attrs["zooms"] = input_metadata['zooms']
            if "data_shape" in input_metadata.keys():
                grp[key].attrs["data_shape"] = input_metadata['data_shape']
            if "bounding_box" in input_metadata.keys():
                grp[key].attrs["bounding_box"] = input_metadata['bounding_box']

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
            grp[key].attrs['gt_filenames'] = input_metadata['gt_filenames']
            grp[key].attrs['data_type'] = gt_metadata['data_type']

            if "zooms" in gt_metadata.keys():
                grp[key].attrs["zooms"] = gt_metadata['zooms']
            if "data_shape" in gt_metadata.keys():
                grp[key].attrs["data_shape"] = gt_metadata['data_shape']
            if gt_metadata['bounding_box'] is not None:
                grp[key].attrs["bounding_box"] = gt_metadata['bounding_box']

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
            grp[key].attrs['roi_filename'] = roi_metadata['gt_filenames']
            grp[key].attrs['data_type'] = roi_metadata['data_type']

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
    """HDF5 dataset object.

    Args:
        root_dir (path): Path of bids and data.
        subject_lst (list of str): List of subjects.
        model_params (dict): Dictionary containing model parameters.
        target_suffix (list of str): List of suffixes of the target structures.
        contrast_params (dict): Dictionary containing contrast parameters.
        slice_axis (int): Indicates the axis used to extract slices: "axial": 2, "sagittal": 0, "coronal": 1.
        transform (Compose): Transformations.
        metadata_choice (str): Choice between "mri_params", "contrasts", None or False, related to FiLM.
        dim (int): Choice 2 or 3, for 2D or 3D data respectively.
        complet (bool): If True removes lines where contrasts is not available.
        slice_filter_fn (SliceFilter): Object that filters slices according to their content.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.
        object_detection_params (dict): Object detection parameters.

    Attributes:
        cst_lst (list): Contrast list.
        gt_lst (list): Contrast label used for ground truth.
        roi_lst (list): Contrast label used for ROI cropping.
        dim (int): Choice 2 or 3, for 2D or 3D data respectively.
        filter_slices (SliceFilter): Object that filters slices according to their content.
        prepro_transforms (Compose): Transforms to be applied before training.
        transform (Compose): Transforms to be applied during training.
        df_object (pd.Dataframe): Dataframe containing dataset information.

    """

    def __init__(self, root_dir, subject_lst, model_params, target_suffix, contrast_params,
                 slice_axis=2, transform=None, metadata_choice=False, dim=2, complet=True,
                 slice_filter_fn=None, roi_params=None, object_detection_params=None, soft_gt=False):
        self.cst_lst = copy.deepcopy(contrast_params["contrast_lst"])
        self.gt_lst = copy.deepcopy(model_params["target_lst"] if "target_lst" in model_params else None)
        self.roi_lst = copy.deepcopy(model_params["roi_lst"] if "roi_lst" in model_params else None)
        self.dim = dim
        self.roi_params = roi_params if roi_params is not None else {"suffix": None, "slice_filter_roi": None}
        self.filter_slices = slice_filter_fn
        self.prepro_transforms, self.transform = transform

        metadata_choice = False if metadata_choice is None else metadata_choice
        # Getting HDS5 dataset file
        if not os.path.exists(model_params["hdf5_path"]):
            print("Computing hdf5 file of the data")
            hdf5_file = Bids_to_hdf5(root_dir,
                                     subject_lst=subject_lst,
                                     hdf5_name=model_params["hdf5_path"],
                                     target_suffix=target_suffix,
                                     roi_params=self.roi_params,
                                     contrast_lst=self.cst_lst,
                                     metadata_choice=metadata_choice,
                                     contrast_balance=contrast_params["balance"],
                                     slice_axis=slice_axis,
                                     slice_filter_fn=slice_filter_fn,
                                     transform=transform,
                                     object_detection_params=object_detection_params,
                                     soft_gt=soft_gt)
            self.hdf5_file = hdf5_file.hdf5_file
        else:
            self.hdf5_file = h5py.File(model_params["hdf5_path"], "r")
        # Loading dataframe object
        self.df_object = Dataframe(self.hdf5_file, self.cst_lst, model_params["csv_path"],
                                   target_suffix=self.gt_lst, roi_suffix=self.roi_lst, dim=self.dim,
                                   filter_slices=slice_filter_fn)
        if complet:
            self.df_object.clean(self.cst_lst)
        print("after cleaning")
        print(self.df_object.df.head())

        self.initial_dataframe = self.df_object.df

        self.dataframe = copy.deepcopy(self.df_object.df)

        self.cst_matrix = np.ones([len(self.dataframe), len(self.cst_lst)], dtype=int)

        # RAM status
        self.status = {ct: False for ct in self.df_object.contrasts}

        ram = model_params["ram"] if "ram" in model_params else True
        if ram:
            self.load_into_ram(self.cst_lst)

    def load_into_ram(self, contrast_lst=None):
        """Aims to load into RAM the contrasts from the list.

        Args:
            contrast_lst (list of str): List of contrasts of interest.
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
        """Set the transforms."""
        self.transform = transform

    def __len__(self):
        """Get the dataset size, ie he number of subvolumes."""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Get samples.

        Warning: For now, this method only supports one gt / roi.

        Args:
            index (int): Sample index.

        Returns:
            dict: Dictionary containing image and label tensors as well as metadata.
        """
        line = self.dataframe.iloc[index]
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

            input_tensors.append(input_tensor)
            # input Metadata
            metadata = imed_loader_utils.SampleMetadata({key: value for key, value in self.hdf5_file['{}/inputs/{}'
                                                        .format(line['Subjects'], ct)].attrs.items()})
            metadata['slice_index'] = line["Slices"]
            metadata['missing_mod'] = missing_modalities
            metadata['crop_params'] = {}
            input_metadata.append(metadata)

        # GT
        gt_img = []
        gt_metadata = []
        for idx, gt in enumerate(self.gt_lst):
            if self.status['gt/' + gt]:
                gt_data = line['gt/' + gt]
            else:
                gt_data = self.hdf5_file[line['gt/' + gt]][line['Slices']]

            gt_data = gt_data.astype(np.uint8)
            gt_img.append(gt_data)
            gt_metadata.append(imed_loader_utils.SampleMetadata({key: value for key, value in
                                                                 self.hdf5_file[line['gt/' + gt]].attrs.items()}))
            gt_metadata[idx]['crop_params'] = {}

        # ROI
        roi_img = []
        roi_metadata = []
        if self.roi_lst:
            if self.status['roi/' + self.roi_lst[0]]:
                roi_data = line['roi/' + self.roi_lst[0]]
            else:
                roi_data = self.hdf5_file[line['roi/' + self.roi_lst[0]]][line['Slices']]

            roi_data = roi_data.astype(np.uint8)
            roi_img.append(roi_data)

            roi_metadata.append(imed_loader_utils.SampleMetadata({key: value for key, value in
                                                                  self.hdf5_file[
                                                                      line['roi/' + self.roi_lst[0]]].attrs.items()}))
            roi_metadata[0]['crop_params'] = {}

        # Run transforms on ROI
        # ROI goes first because params of ROICrop are needed for the followings
        stack_roi, metadata_roi = self.transform(sample=roi_img,
                                                 metadata=roi_metadata,
                                                 data_type="roi")
        # Update metadata_input with metadata_roi
        metadata_input = imed_loader_utils.update_metadata(metadata_roi, input_metadata)

        # Run transforms on images
        stack_input, metadata_input = self.transform(sample=input_tensors,
                                                     metadata=metadata_input,
                                                     data_type="im")
        # Update metadata_input with metadata_roi
        metadata_gt = imed_loader_utils.update_metadata(metadata_input, gt_metadata)

        # Run transforms on images
        stack_gt, metadata_gt = self.transform(sample=gt_img,
                                               metadata=metadata_gt,
                                               data_type="gt")
        data_dict = {
            'input': stack_input,
            'gt': stack_gt,
            'roi': stack_roi,
            'input_metadata': metadata_input,
            'gt_metadata': metadata_gt,
            'roi_metadata': metadata_roi
        }

        return data_dict

    def update(self, strategy="Missing", p=0.0001):
        """Update the Dataframe itself.

        Args:
            p (float): Float between 0 and 1, probability of the contrast to be missing.
            strategy (str): Update the dataframe using the corresponding strategy. For now the only the strategy
                implemented is the one used by HeMIS (i.e. by removing contrasts with a certain probability.) Other
                strategies that could be implemented are Active Learning, Curriculum Learning, ...
        """
        if strategy == 'Missing':
            print("Probalility of missing contrast = {}".format(p))
            for idx in range(len(self.dataframe)):
                missing_mod = np.random.choice(2, len(self.cst_lst), p=[p, 1 - p])
                # if all contrasts are removed from a subject randomly choose 1
                if not np.any(missing_mod):
                    missing_mod = np.zeros((len(self.cst_lst)))
                    missing_mod[np.random.randint(2, size=1)] = 1
                self.cst_matrix[idx,] = missing_mod

            print("Missing contrasts = {}".format(self.cst_matrix.size - self.cst_matrix.sum()))


def HDF5_to_Bids(HDF5, subjects, path_dir):
    """Convert HDF5 file to BIDS dataset.

    Args:
        HDF5 (str): Path to the HDF5 file.
        subjects (list): List of subject names.
        path_dir (str): Output folder path, already existing.
    """
    # Open FDH5 file
    hdf5 = h5py.File(HDF5, "r")
    # check the dir exists:
    if not path.exists(path_dir):
        raise FileNotFoundError("Directory {} doesn't exist. Stopping process.".format(path_dir))

    # loop over all subjects
    for sub in subjects:
        path_sub = path_dir + '/' + sub + '/anat/'
        path_label = path_dir + '/derivatives/labels/' + sub + '/anat/'

        if not path.exists(path_sub):
            os.makedirs(path_sub)

        if not path.exists(path_label):
            os.makedirs(path_label)

        # Get Subject Group
        try:
            grp = hdf5[sub]
        except:
            continue
        # inputs
        cts = grp['inputs'].attrs['contrast']

        # Relation between voxel and world coordinates is not available
        for ct in cts:
            input_data = np.array(grp['inputs/{}'.format(ct)])
            nib_image = nib.Nifti1Image(input_data, np.eye(4))
            filename = os.path.join(path_sub, sub + "_" + ct + ".nii.gz")
            nib.save(nib_image, filename)

        # GT
        cts = grp['gt'].attrs['contrast']

        for ct in cts:
            for filename in grp['gt/{}'.format(ct)].attrs['gt_filenames']:
                gt_data = grp['gt/{}'.format(ct)]
                nib_image = nib.Nifti1Image(gt_data, np.eye(4))
                filename = os.path.join(path_label, filename.split("/")[-1])
                nib.save(nib_image, filename)

        cts = grp['roi'].attrs['contrast']

        for ct in cts:
            roi_data = grp['roi/{}'.format(ct)]
            if np.any(roi_data.shape):
                nib_image = nib.Nifti1Image(roi_data, np.eye(4))
                filename = os.path.join(path_label, grp['roi/{}'.format(ct)].attrs['roi_filename'][0].split("/")[-1])
                nib.save(nib_image, filename)

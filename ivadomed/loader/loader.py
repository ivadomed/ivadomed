import nibabel as nib
import numpy as np
import torch
from bids_neuropoly import bids
from torch.utils.data import Dataset
from tqdm import tqdm

from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, adaptative as imed_adaptative, film as imed_film
from ivadomed import preprocessing as imed_prepro


def load_dataset(data_list, bids_path, transforms_params, model_params, target_suffix, roi_params,
                 contrast_params, slice_filter_params, slice_axis, multichannel,
                 dataset_type="training", requires_undo=False, metadata_type=None, **kwargs):
    """Get loader.

    Args:
        data_list (list):
        bids_path (string):
        transforms_params (dict):
        model_name (string):
        target_suffix (list):
        roi_params (dict):
        contrast_params (dict):
        slice_filter_params (dict):
        slice_axis (string):
        multichannel (bool):
        metadata_type (string): None if no metadata
        dataset_type (string): training, validation or testing
        requires_undo (Bool): If True, the transformations without undo_transform will be discarded
    Returns:
        BidsDataset
    """
    # Compose transforms
    preprocessing_transforms = imed_prepro.get_preprocessing_transforms(transforms_params)
    prepro_transforms = imed_transforms.Compose(preprocessing_transforms, requires_undo=requires_undo)
    transforms = imed_transforms.Compose(transforms_params, requires_undo=requires_undo)
    tranform_lst = [prepro_transforms if len(preprocessing_transforms) else None, transforms]

    if model_params["name"] == "UNet3D":
        dataset = Bids3DDataset(bids_path,
                                subject_lst=data_list,
                                target_suffix=target_suffix,
                                roi_suffix=roi_params["suffix"],
                                contrast_params=contrast_params,
                                metadata_choice=metadata_type,
                                slice_axis=imed_utils.AXIS_DCT[slice_axis],
                                transform=tranform_lst,
                                multichannel=multichannel,
                                model_params=model_params)
    elif model_params["name"] == "HeMISUnet":
        dataset = imed_adaptative.HDF5Dataset(root_dir=bids_path,
                                              subject_lst=data_list,
                                              model_params=model_params,
                                              contrast_params=contrast_params,
                                              target_suffix=target_suffix,
                                              slice_axis=imed_utils.AXIS_DCT[slice_axis],
                                              transform=tranform_lst,
                                              metadata_choice=metadata_type,
                                              slice_filter_fn=imed_utils.SliceFilter(**slice_filter_params),
                                              roi_suffix=roi_params["suffix"])
    else:
        dataset = BidsDataset(bids_path,
                              subject_lst=data_list,
                              target_suffix=target_suffix,
                              roi_suffix=roi_params["suffix"],
                              contrast_params=contrast_params,
                              metadata_choice=metadata_type,
                              slice_axis=imed_utils.AXIS_DCT[slice_axis],
                              transform=tranform_lst,
                              multichannel=multichannel,
                              slice_filter_fn=imed_utils.SliceFilter(**slice_filter_params))
        dataset.load_filenames()

    # if ROICrop in transform, then apply SliceFilter to ROI slices
    if 'ROICrop' in transforms_params:
        dataset = imed_loader_utils.filter_roi(dataset, nb_nonzero_thr=roi_params["slice_filter_roi"])

    if model_params["name"] != "UNet3D":
        print("Loaded {} {} slices for the {} set.".format(len(dataset), slice_axis, dataset_type))
    else:
        print("Loaded {} volumes of size {} for the {} set.".format(len(dataset), slice_axis, dataset_type))

    return dataset


class SegmentationPair(object):
    """This class is used to build segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).

    :param input_filenames: the input filename list (supported by nibabel). For single channel, the list will contain 1
                           input filename.
    :param gt_filenames: the ground-truth filenames list.
    :param metadata: metadata list with each item corresponding to an image (modality) in input_filenames.  For single
                     channel, the list will contain metadata related to one image.
    :param cache: if the data should be cached in memory or not.
    """

    def __init__(self, input_filenames, gt_filenames, metadata=None, slice_axis=2, cache=True,
                 prepro_transforms=None):

        self.input_filenames = input_filenames
        self.gt_filenames = gt_filenames
        self.metadata = metadata
        self.cache = cache
        self.slice_axis = slice_axis
        self.prepro_transforms = prepro_transforms

        # list of the images
        self.input_handle = []

        # loop over the filenames (list)
        for input_file in self.input_filenames:
            input_img = nib.load(input_file)
            self.input_handle.append(input_img)
            if len(input_img.shape) > 3:
                raise RuntimeError("4-dimensional volumes not supported.")

        # list of GT for multiclass segmentation
        self.gt_handle = []

        # Unlabeled data (inference time)
        if self.gt_filenames is not None:
            for gt in self.gt_filenames:
                if gt is not None:
                    self.gt_handle.append(nib.load(gt))
                else:
                    self.gt_handle.append(None)

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_filenames is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        for idx, handle in enumerate(self.input_handle):
            self.input_handle[idx] = nib.as_closest_canonical(handle)

        # Unlabeled data
        if self.gt_filenames is not None:
            for idx, gt in enumerate(self.gt_handle):
                if gt is not None:
                    self.gt_handle[idx] = nib.as_closest_canonical(gt)

        if self.metadata:
            self.metadata = []
            for data, input_filename in zip(metadata, input_filenames):
                data["input_filenames"] = input_filename
                data["gt_filenames"] = gt_filenames
                self.metadata.append(data)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = []
        for handle in self.input_handle:
            shape = imed_loader_utils.orient_shapes_hwd(handle.header.get_data_shape(), self.slice_axis)
            input_shape.append(tuple(shape))

            if not len(set(input_shape)):
                raise RuntimeError('Inputs have different dimensions.')

        gt_shape = []

        for gt in self.gt_handle:
            if gt is not None:
                shape = imed_loader_utils.orient_shapes_hwd(gt.header.get_data_shape(), self.slice_axis)
                gt_shape.append(tuple(shape))

                if not len(set(gt_shape)):
                    raise RuntimeError('Labels have different dimensions.')

        return input_shape[0], gt_shape[0] if len(gt_shape) else None

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'

        input_data = []
        for handle in self.input_handle:
            hwd_oriented = imed_loader_utils.orient_img_hwd(handle.get_fdata(cache_mode, dtype=np.float32),
                                                            self.slice_axis)
            input_data.append(hwd_oriented)

        gt_data = []
        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        for gt in self.gt_handle:
            if gt is not None:
                hwd_oriented = imed_loader_utils.orient_img_hwd(gt.get_fdata(cache_mode, dtype=np.float32),
                                                                self.slice_axis)
                gt_data.append(hwd_oriented.astype(np.uint8))
            else:
                gt_data.append(np.zeros(self.input_handle[0].shape, dtype=np.float32).astype(np.uint8))

        return input_data, gt_data

    def get_pair_metadata(self, slice_index=0):
        gt_meta_dict = []
        for gt in self.gt_handle:
            if gt is not None:
                gt_meta_dict.append(imed_loader_utils.SampleMetadata({
                    "zooms": imed_loader_utils.orient_shapes_hwd(gt.header.get_zooms(), self.slice_axis),
                    "data_shape": imed_loader_utils.orient_shapes_hwd(gt.header.get_data_shape(), self.slice_axis),
                    "gt_filenames": self.metadata[0]["gt_filenames"],
                    "data_type": 'gt'
                }))
            else:
                gt_meta_dict.append(gt_meta_dict[0])

        input_meta_dict = []
        for handle in self.input_handle:
            input_meta_dict.append(imed_loader_utils.SampleMetadata({
                "zooms": imed_loader_utils.orient_shapes_hwd(handle.header.get_zooms(), self.slice_axis),
                "data_shape": imed_loader_utils.orient_shapes_hwd(handle.header.get_data_shape(), self.slice_axis),
                "data_type": 'im'
            }))

        dreturn = {
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }

        for idx, metadata in enumerate(self.metadata):  # loop across channels
            metadata["slice_index"] = slice_index
            self.metadata[idx] = metadata
            for metadata_key in metadata.keys():  # loop across input metadata
                dreturn["input_metadata"][idx][metadata_key] = metadata[metadata_key]

        return dreturn

    def get_pair_slice(self, slice_index):
        """Return the specified slice from (input, ground truth).

        :param slice_index: the slice number
        """

        metadata = self.get_pair_metadata(slice_index)
        input_dataobj, gt_dataobj = self.get_pair_data()

        if self.slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        input_slices = []
        # Loop over modalities
        for data_object in input_dataobj:
            input_slices.append(np.asarray(data_object[..., slice_index],
                                           dtype=np.float32))

        # Handle the case for unlabeled data
        if self.gt_handle is None:
            gt_slices = None
        else:
            gt_slices = []
            for gt_obj in gt_dataobj:
                gt_slices.append(np.asarray(gt_obj[..., slice_index],
                                            dtype=np.float32))

        dreturn = {
            "input": input_slices,
            "gt": gt_slices,
            "input_metadata": metadata["input_metadata"],
            "gt_metadata": metadata["gt_metadata"],
        }

        return dreturn


class MRI2DSegmentationDataset(Dataset):
    """This is a generic class for 2D (slice-wise) segmentation datasets.

    :param filename_pairs: a list of tuples in the format (input filename list containing all modalities,
                           ground truth filename, ROI filename, metadata).
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """

    def __init__(self, filename_pairs, slice_axis=2, cache=True, transform=None, slice_filter_fn=None):
        self.indexes = []
        self.filename_pairs = filename_pairs
        if isinstance(transform, list):
            self.prepro_transforms, self.transform = transform
        else:
            self.transform = transform
            self.prepro_transforms = None
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.n_contrasts = len(self.filename_pairs[0][0])

    def load_filenames(self):
        for input_filenames, gt_filenames, roi_filename, metadata in self.filename_pairs:
            roi_pair = SegmentationPair(input_filenames, roi_filename, metadata=metadata, slice_axis=self.slice_axis,
                                        cache=self.cache, prepro_transforms=self.prepro_transforms)

            seg_pair = SegmentationPair(input_filenames, gt_filenames, metadata=metadata, slice_axis=self.slice_axis,
                                        cache=self.cache, prepro_transforms=self.prepro_transforms)

            input_data_shape, _ = seg_pair.get_pair_shapes()

            for idx_pair_slice in range(input_data_shape[-1]):
                slice_seg_pair = seg_pair.get_pair_slice(idx_pair_slice)
                if self.slice_filter_fn:
                    filter_fn_ret_seg = self.slice_filter_fn(slice_seg_pair)
                if self.slice_filter_fn and not filter_fn_ret_seg:
                    continue

                slice_roi_pair = roi_pair.get_pair_slice(idx_pair_slice)

                item = imed_prepro.apply_transforms(self.prepro_transforms, slice_seg_pair, slice_roi_pair)
                self.indexes.append(item)

    def set_transform(self, transform):
        """ This method will replace the current transformation for the
        dataset.

        :param transform: the new transformation
        """
        self.transform = transform

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index (input, ground truth, roi and metadatas).

        :param index: slice index.
        """
        seg_pair_slice, roi_pair_slice = self.indexes[index]

        # Run transforms on ROI
        # ROI goes first because params of ROICrop are needed for the followings
        stack_roi, metadata_roi = self.transform(sample=roi_pair_slice["gt"],
                                                 metadata=roi_pair_slice['gt_metadata'],
                                                 data_type="roi")
        # Update metadata_input with metadata_roi
        metadata_input = imed_loader_utils.update_metadata(metadata_roi, seg_pair_slice['input_metadata'])

        # Run transforms on images
        stack_input, metadata_input = self.transform(sample=seg_pair_slice["input"],
                                                     metadata=metadata_input,
                                                     data_type="im")

        # Update metadata_input with metadata_roi
        metadata_gt = imed_loader_utils.update_metadata(metadata_input, seg_pair_slice['gt_metadata'])

        # Run transforms on images
        stack_gt, metadata_gt = self.transform(sample=seg_pair_slice["gt"],
                                               metadata=metadata_gt,
                                               data_type="gt")
        # Make sure stack_gt is binarized
        if stack_gt is not None:
            stack_gt = torch.as_tensor(
                [imed_postpro.threshold_predictions(stack_gt[i_label, :], thr=0.1) for i_label in range(len(stack_gt))])

        data_dict = {
            'input': stack_input,
            'gt': stack_gt,
            'roi': stack_roi,
            'input_metadata': metadata_input,
            'gt_metadata': metadata_gt,
            'roi_metadata': metadata_roi
        }

        return data_dict


class MRI3DSubVolumeSegmentationDataset(Dataset):
    """This is a generic class for 3D segmentation datasets. This class overload
    MRI3DSegmentationDataset by splitting the initials volumes in several
    subvolumes. Each subvolumes will be of the sizes of the length parameter.

    This class also implement a padding parameter, which overlap the borders of
    the different (the borders of the upper-volume aren't superposed). For
    example if you have a length of (32,32,32) and a padding of 16, your final
    subvolumes will have a total lengths of (64,64,64) with the voxels contained
    outside the core volume and which are shared with the other subvolumes.

    Be careful, the input's dimensions should be compatible with the given
    lengths and paddings. This class doesn't handle missing dimensions.

    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    :param length: size of each dimensions of the subvolumes
    :param padding: size of the overlapping per subvolume and dimensions
    """

    def __init__(self, filename_pairs, transform, length=(64, 64, 64), padding=0, slice_axis=0):
        self.filename_pairs = filename_pairs
        self.handlers = []
        self.indexes = []
        self.length = length
        self.padding = padding
        if isinstance(transform, list):
            self.prepro_transforms, self.transform = transform
        else:
            self.transform = transform
            self.prepro_transforms = None
        self.slice_axis = slice_axis

        self._load_filenames()
        self._prepare_indices()

    def _load_filenames(self):
        for input_filename, gt_filename, roi_filename, metadata in self.filename_pairs:
            segpair = SegmentationPair(input_filename, gt_filename, metadata=metadata, slice_axis=self.slice_axis)
            input_data, gt_data = segpair.get_pair_data()
            metadata = segpair.get_pair_metadata()
            seg_pair = {
                'input': input_data,
                'gt': gt_data,
                'input_metadata': metadata['input_metadata'],
                'gt_metadata': metadata['gt_metadata']
            }

            self.handlers.append(imed_prepro.apply_transforms(self.prepro_transforms, seg_pair=seg_pair))

    def _prepare_indices(self):
        length = self.length
        padding = self.padding

        crop = False

        for idx, transfo in enumerate(self.transform.transform["im"].transforms):
            if "CenterCrop" in str(type(transfo)):
                crop = True
                shape_crop = transfo.size

        for i in range(0, len(self.handlers)):
            if not crop:
                input_img = self.handlers[i][0]['input']
                shape = input_img[0].shape
            else:
                shape = shape_crop
            if (shape[0] - 2 * padding) % length[0] != 0 or shape[0] % 16 != 0 \
                    or (shape[1] - 2 * padding) % length[1] != 0 or shape[1] % 16 != 0 \
                    or (shape[2] - 2 * padding) % length[2] != 0 or shape[2] % 16 != 0:
                raise RuntimeError('Input shape of each dimension should be a \
                                    multiple of length plus 2 * padding and a multiple of 16.')

            for x in range(length[0] + padding, shape[0] - padding + 1, length[0]):
                for y in range(length[1] + padding, shape[1] - padding + 1, length[1]):
                    for z in range(length[2] + padding, shape[2] - padding + 1, length[2]):
                        self.indexes.append({
                            'x_min': x - length[0] - padding,
                            'x_max': x + padding,
                            'y_min': y - length[1] - padding,
                            'y_max': y + padding,
                            'z_min': z - length[2] - padding,
                            'z_max': z + padding,
                            'handler_index': i})

    def __len__(self):
        """Return the dataset size. The number of subvolumes."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair subvolume (input, ground truth).

        :param index: subvolume index.
        """
        coord = self.indexes[index]
        seg_pair, _ = self.handlers[coord['handler_index']]

        # Run transforms on images
        stack_input, metadata_input = self.transform(sample=seg_pair['input'],
                                                     metadata=seg_pair['input_metadata'],
                                                     data_type="im")
        # Update metadata_gt with metadata_input
        metadata_gt = imed_loader_utils.update_metadata(metadata_input, seg_pair['gt_metadata'])

        # Run transforms on images
        stack_gt, metadata_gt = self.transform(sample=seg_pair['gt'],
                                               metadata=metadata_gt,
                                               data_type="gt")
        # Make sure stack_gt is binarized
        if stack_gt is not None:
            stack_gt = torch.as_tensor(
                [imed_postpro.threshold_predictions(stack_gt[i_label, :], thr=0.1) for i_label in range(len(stack_gt))])

        shape_x = coord["x_max"] - coord["x_min"]
        shape_y = coord["y_max"] - coord["y_min"]
        shape_z = coord["z_max"] - coord["z_min"]

        subvolumes = {
            'input': torch.zeros(stack_input.shape[0], shape_x, shape_y, shape_z),
            'gt': torch.zeros(stack_input.shape[0], shape_x, shape_y, shape_z),
            'input_metadata': metadata_input,
            'gt_metadata': metadata_gt
        }

        for _ in range(len(stack_input)):
            subvolumes['input'] = stack_input[:,
                                  coord['x_min']:coord['x_max'],
                                  coord['y_min']:coord['y_max'],
                                  coord['z_min']:coord['z_max']]

        if stack_gt is not None:
            for _ in range(len(stack_gt)):
                subvolumes['gt'] = stack_gt[:,
                                   coord['x_min']:coord['x_max'],
                                   coord['y_min']:coord['y_max'],
                                   coord['z_min']:coord['z_max']]

        subvolumes['gt'] = subvolumes['gt'].type(torch.BoolTensor)
        return subvolumes


class Bids3DDataset(MRI3DSubVolumeSegmentationDataset):
    def __init__(self, root_dir, subject_lst, target_suffix, model_params, contrast_params, slice_axis=2,
                 cache=True, transform=None, metadata_choice=False, roi_suffix=None,
                 multichannel=False):
        dataset = BidsDataset(root_dir,
                              subject_lst=subject_lst,
                              target_suffix=target_suffix,
                              roi_suffix=roi_suffix,
                              contrast_params=contrast_params,
                              metadata_choice=metadata_choice,
                              slice_axis=slice_axis,
                              transform=transform,
                              multichannel=multichannel)

        super().__init__(dataset.filename_pairs, length=model_params["length_3D"], padding=model_params["padding_3D"],
                         transform=transform, slice_axis=slice_axis)


class BidsDataset(MRI2DSegmentationDataset):
    def __init__(self, root_dir, subject_lst, target_suffix, contrast_params, slice_axis=2,
                 cache=True, transform=None, metadata_choice=False, slice_filter_fn=None, roi_suffix=None,
                 multichannel=False):

        self.bids_ds = bids.BIDS(root_dir)

        self.filename_pairs = []
        if metadata_choice == 'mri_params':
            self.metadata = {"FlipAngle": [], "RepetitionTime": [],
                             "EchoTime": [], "Manufacturer": []}

        bids_subjects = [s for s in self.bids_ds.get_subjects() if s.record["subject_id"] in subject_lst]

        # Create a list with the filenames for all contrasts and subjects
        subjects_tot = []
        for subject in bids_subjects:
            subjects_tot.append(str(subject.record["absolute_path"]))

        # Create a dictionary with the number of subjects for each contrast of contrast_balance

        tot = {contrast: len([s for s in bids_subjects if s.record["modality"] == contrast])
               for contrast in contrast_params["balance"].keys()}

        # Create a counter that helps to balance the contrasts
        c = {contrast: 0 for contrast in contrast_params["balance"].keys()}

        multichannel_subjects = {}
        if multichannel:
            num_contrast = len(contrast_params["contrast_lst"])
            idx_dict = {}
            for idx, contrast in enumerate(contrast_params["contrast_lst"]):
                idx_dict[contrast] = idx
            multichannel_subjects = {subject: {"absolute_paths": [None] * num_contrast,
                                               "deriv_path": None,
                                               "roi_filename": None,
                                               "metadata": [None] * num_contrast} for subject in subject_lst}

        for subject in tqdm(bids_subjects, desc="Loading dataset"):
            if subject.record["modality"] in contrast_params["contrast_lst"]:
                # Training & Validation: do not consider the contrasts over the threshold contained in contrast_balance
                if subject.record["modality"] in contrast_params["balance"].keys():
                    c[subject.record["modality"]] = c[subject.record["modality"]] + 1
                    if c[subject.record["modality"]] / tot[subject.record["modality"]] > contrast_params["balance"][
                        subject.record["modality"]]:
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

                    if not (roi_suffix is None) and \
                            deriv.endswith(subject.record["modality"] + roi_suffix + ".nii.gz"):
                        roi_filename = [deriv]

                if (not any(target_filename)) or (not (roi_suffix is None) and (roi_filename is None)):
                    continue

                if not subject.has_metadata():
                    metadata = {}
                else:
                    metadata = subject.metadata()

                # add contrast to metadata
                metadata['contrast'] = subject.record["modality"]

                if metadata_choice == 'mri_params':
                    if not all([imed_film.check_isMRIparam(m, metadata, subject, self.metadata) for m in
                                self.metadata.keys()]):
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
                    self.filename_pairs.append(([subject.record.absolute_path],
                                                target_filename, roi_filename, [metadata]))

        if multichannel:
            for subject in multichannel_subjects.values():
                if not None in subject["absolute_paths"]:
                    self.filename_pairs.append((subject["absolute_paths"], subject["deriv_path"],
                                                subject["roi_filename"], subject["metadata"]))

        super().__init__(self.filename_pairs, slice_axis, cache, transform, slice_filter_fn)

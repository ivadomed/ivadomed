import functools
import math
import numbers
import random

import numpy as np
import torch
from scipy.ndimage import rotate, shift
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.measurements import label, center_of_mass
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes, binary_closing
from skimage.exposure import equalize_adapthist
from torchvision import transforms as torchvision_transforms


def multichannel_capable(wrapped):
    @functools.wraps(wrapped)
    def wrapper(self, sample, metadata):
        if isinstance(sample, list):
            list_data, list_metadata = [], []
            for s_cur, m_cur in zip(sample, metadata):
                # Run function for each sample of the list
                data_cur, metadata_cur = wrapped(self, s_cur, m_cur)
                list_data.append(data_cur)
                list_metadata.append(metadata_cur)
            return list_data, list_metadata
        # If sample is None, then return a pair (None, None)
        if sample is None:
            return None, None
        else:
            return wrapped(self, sample, metadata)

    return wrapper


def two_dim_compatible(wrapped):
    @functools.wraps(wrapped)
    def wrapper(self, sample, metadata):
        # Check if sample is 2D
        if len(sample.shape) == 2:
            # Add one dimension
            sample = np.expand_dims(sample, axis=-1)
            # Run transform
            result_sample, result_metadata = wrapped(self, sample, metadata)
            # Remove last dimension
            return np.squeeze(result_sample, axis=-1), result_metadata
        else:
            return wrapped(self, sample, metadata)

    return wrapper


class ImedTransform(object):

    def __call__(self, sample, metadata=None):
        raise NotImplementedError("You need to implement the transform() method.")

    def undo_transform(self, sample, metadata=None):
        raise NotImplementedError("You need to implement the undo_transform() method.")


class Compose(object):
    """Composes transforms together.

    Composes transforms together and split between images, GT and ROI.

    self.transform is a dict:
        - keys: "im", "gt" and "roi"
        - values torchvision_transform.Compose objects.

    Attributes:
        dict_transforms (dictionary): Dictionary where the keys are the transform names
            and the value their parameters.
        requires_undo (bool): If True, does not include transforms which do not have an undo_transform
            implemented yet.
    """

    def __init__(self, dict_transforms, requires_undo=False):
        list_tr_im, list_tr_gt, list_tr_roi = [], [], []
        for transform in dict_transforms.keys():
            parameters = dict_transforms[transform]

            # Get list of data type
            if "applied_to" in parameters:
                list_applied_to = parameters["applied_to"]
                del parameters['applied_to']
            else:
                list_applied_to = ["im", "gt", "roi"]

            # call transform
            if transform in globals():
                transform_obj = globals()[transform](**parameters)
            else:
                print('ERROR: {} transform is not available in your ivadomed package. '
                      'Please check its compatibility with your model json file.'.format(transform))
                exit()

            # check if undo_transform method is implemented
            if requires_undo:
                if not hasattr(transform_obj, 'undo_transform'):
                    print('{} transform not included since no undo_transform available for it.'.format(transform))
                    continue

            if "im" in list_applied_to:
                list_tr_im.append(transform_obj)
            if "roi" in list_applied_to:
                list_tr_roi.append(transform_obj)
            if "gt" in list_applied_to:
                list_tr_gt.append(transform_obj)

        self.transform = {
            "im": torchvision_transforms.Compose(list_tr_im),
            "gt": torchvision_transforms.Compose(list_tr_gt),
            "roi": torchvision_transforms.Compose(list_tr_roi)}

    def __call__(self, sample, metadata, data_type='im'):
        if self.transform[data_type] is None or len(metadata) == 0:
            # In case self.transform[data_type] is None
            return None, None
        else:
            for tr in self.transform[data_type].transforms:
                sample, metadata = tr(sample, metadata)
            return sample, metadata


class UndoCompose(object):
    def __init__(self, compose):
        self.transforms = compose

    def __call__(self, sample, metadata, data_type='gt'):
        if self.transforms.transform[data_type] is None:
            # In case self.transforms.transform[data_type] is None
            return None, None
        else:
            for tr in self.transforms.transform[data_type].transforms[::-1]:
                sample, metadata = tr.undo_transform(sample, metadata)
            return sample, metadata


class UndoTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform.undo_transform(sample)


class NumpyToTensor(ImedTransform):
    """Converts numpy array to tensor object."""

    def undo_transform(self, sample, metadata=None):
        return list(sample.numpy()), metadata

    def __call__(self, sample, metadata=None):
        sample = np.array(sample)
        # Use np.ascontiguousarray to avoid axes permutations issues
        arr_contig = np.ascontiguousarray(sample, dtype=sample.dtype)
        return torch.from_numpy(arr_contig), metadata


class Resample(ImedTransform):
    """
    Resample image to a given resolution.

    Args:
        hspace (float): Resolution along the first axis, in mm.
        wspace (float): Resolution along the second axis, in mm.
        dspace (float): Resolution along the third axis, in mm.
        interpolation_order (int): Order of spline interpolation. Set to 0 for label data. Default=2.
    """

    def __init__(self, hspace, wspace, dspace=1.):
        self.hspace = hspace
        self.wspace = wspace
        self.dspace = dspace

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        assert "data_shape" in metadata
        is_2d = sample.shape[-1] == 1

        # Get params
        original_shape = metadata["data_shape"]
        current_shape = sample.shape
        params_undo = [x / y for x, y in zip(original_shape, current_shape)]
        if is_2d:
            params_undo[-1] = 1.0

        # Undo resampling
        data_out = zoom(sample,
                        zoom=params_undo,
                        order=0 if metadata['data_type'] == 'gt' else 2)

        # Data type
        data_out = data_out.astype(sample.dtype)

        return data_out, metadata

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # Get params
        # Voxel dimension in mm
        is_2d = sample.shape[-1] == 1
        zooms = list(metadata["zooms"])

        if len(zooms) == 2:
            zooms += [1.0]

        hfactor = zooms[0] / self.hspace
        wfactor = zooms[1] / self.wspace
        dfactor = zooms[2] / self.dspace
        params_resample = (hfactor, wfactor, dfactor) if not is_2d else (hfactor, wfactor, 1.0)

        # Run resampling
        data_out = zoom(sample,
                        zoom=params_resample,
                        order=0 if metadata['data_type'] == 'gt' else 2)

        # Data type
        data_out = data_out.astype(sample.dtype)

        return data_out, metadata


class NormalizeInstance(ImedTransform):
    """Normalize a tensor or an array image with mean and standard deviation estimated
    from the sample itself.
    """

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        data_out = (sample - sample.mean()) / sample.std()
        return data_out, metadata


class CroppableArray(np.ndarray):
    """Adapted From: https://stackoverflow.com/a/41155020/13306686"""

    def __getitem__(self, item):
        all_in_slices = []
        pad = []
        for dim in range(self.ndim):
            # If the slice has no length then it's a single argument.
            # If it's just an integer then we just return, this is
            # needed for the representation to work properly
            # If it's not then create a list containing None-slices
            # for dim>=1 and continue down the loop
            try:
                len(item)
            except TypeError:
                if isinstance(item, int):
                    return super().__getitem__(item)
                newitem = [slice(None)] * self.ndim
                newitem[0] = item
                item = newitem
            # We're out of items, just append noop slices
            if dim >= len(item):
                all_in_slices.append(slice(0, self.shape[dim]))
                pad.append((0, 0))
            # We're dealing with an integer (no padding even if it's
            # out of bounds)
            if isinstance(item[dim], int):
                all_in_slices.append(slice(item[dim], item[dim] + 1))
                pad.append((0, 0))
            # Dealing with a slice, here it get's complicated, we need
            # to correctly deal with None start/stop as well as with
            # out-of-bound values and correct padding
            elif isinstance(item[dim], slice):
                # Placeholders for values
                start, stop = 0, self.shape[dim]
                this_pad = [0, 0]
                if item[dim].start is None:
                    start = 0
                else:
                    if item[dim].start < 0:
                        this_pad[0] = -item[dim].start
                        start = 0
                    else:
                        start = item[dim].start
                if item[dim].stop is None:
                    stop = self.shape[dim]
                else:
                    if item[dim].stop > self.shape[dim]:
                        this_pad[1] = item[dim].stop - self.shape[dim]
                        stop = self.shape[dim]
                    else:
                        stop = item[dim].stop
                all_in_slices.append(slice(start, stop))
                pad.append(tuple(this_pad))

        # Let numpy deal with slicing
        ret = super().__getitem__(tuple(all_in_slices))
        # and padding
        ret = np.pad(ret, tuple(pad), mode='constant', constant_values=0)

        return ret


class Crop(ImedTransform):
    def __init__(self, size):
        self.size = size if len(size) == 3 else size + [0]

    @staticmethod
    def _adjust_padding(npad, sample):
        npad_out_tuple = []
        for idx_dim, tuple_pad in enumerate(npad):
            pad_start, pad_end = tuple_pad
            if pad_start < 0 or pad_end < 0:
                # Move axis of interest
                sample_reorient = np.swapaxes(sample, 0, idx_dim)
                # Adjust pad and crop
                if pad_start < 0 and pad_end < 0:
                    sample_crop = sample_reorient[abs(pad_start):pad_end, ]
                    pad_end, pad_start = 0, 0
                elif pad_start < 0:
                    sample_crop = sample_reorient[abs(pad_start):, ]
                    pad_start = 0
                else:  # i.e. pad_end < 0:
                    sample_crop = sample_reorient[:pad_end, ]
                    pad_end = 0
                # Reorient
                sample = np.swapaxes(sample_crop, 0, idx_dim)

            npad_out_tuple.append((pad_start, pad_end))

        return npad_out_tuple, sample

    @multichannel_capable
    def __call__(self, sample, metadata):
        # Get params
        is_2d = sample.shape[-1] == 1
        th, tw, td = self.size
        fh, fw, fd, h, w, d = metadata['crop_params']

        # Crop data
        # Note we use here CroppableArray in order to deal with "out of boundaries" crop
        # e.g. if fh is negative or fh+th out of bounds, then it will pad
        if is_2d:
            data_out = sample.view(CroppableArray)[fh:fh + th, fw:fw + tw, :]
        else:
            data_out = sample.view(CroppableArray)[fh:fh + th, fw:fw + tw, fd:fd + td]

        return data_out, metadata

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        # Get crop params
        is_2d = sample.shape[-1] == 1
        th, tw, td = self.size
        fh, fw, fd, h, w, d = metadata["crop_params"]

        # Compute params to undo transform
        pad_left = fw
        pad_right = w - pad_left - tw
        pad_top = fh
        pad_bottom = h - pad_top - th
        pad_front = fd if not is_2d else 0
        pad_back = d - pad_front - td if not is_2d else 0
        npad = [(pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)]

        # Check and adjust npad if needed, i.e. if crop out of boundaries
        npad_adj, sample_adj = self._adjust_padding(npad, sample.copy())

        # Apply padding
        data_out = np.pad(sample_adj,
                          npad_adj,
                          mode='constant',
                          constant_values=0).astype(sample.dtype)

        return data_out, metadata


class CenterCrop(Crop):
    """Make a centered crop of a specified size."""

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # Crop parameters
        th, tw, td = self.size
        h, w, d = sample.shape
        fh = int(round((h - th) / 2.))
        fw = int(round((w - tw) / 2.))
        fd = int(round((d - td) / 2.))
        params = (fh, fw, fd, h, w, d)
        metadata['crop_params'] = params

        # Call base method
        return super().__call__(sample, metadata)


class ROICrop(Crop):
    """Make a crop of a specified size around a ROI."""

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # If crop_params are not in metadata,
        # then we are here dealing with ROI data to determine crop params
        if 'crop_params' not in metadata:
            # Compute center of mass of the ROI
            h_roi, w_roi, d_roi = center_of_mass(sample.astype(np.int))
            h_roi, w_roi, d_roi = int(round(h_roi)), int(round(w_roi)), int(round(d_roi))
            th, tw, td = self.size
            th_half, tw_half, td_half = int(round(th / 2.)), int(round(tw / 2.)), int(round(td / 2.))

            # compute top left corner of the crop area
            fh = h_roi - th_half
            fw = w_roi - tw_half
            fd = d_roi - td_half

            # Crop params
            h, w, d = sample.shape
            params = (fh, fw, fd, h, w, d)
            metadata['crop_params'] = params

        # Call base method
        return super().__call__(sample, metadata)


class DilateGT(ImedTransform):
    """Randomly dilate a tensor ground-truth.
    :param dilation_factor: float, controls the number of dilation iterations.
                            For each individual lesion, the number of dilation iterations is computed as follows:
                                nb_it = int(round(dilation_factor * sqrt(lesion_area)))
                            If dilation_factor <= 0, then no dilation will be perfomed.
    """

    def __init__(self, dilation_factor):
        self.dil_factor = dilation_factor

    @staticmethod
    def dilate_lesion(arr_bin, arr_soft, label_values):
        for lb in label_values:
            # binary dilation with 1 iteration
            arr_dilated = binary_dilation(arr_bin, iterations=1)

            # isolate new voxels, i.e. the ones from the dilation
            new_voxels = np.logical_xor(arr_dilated, arr_bin).astype(np.int)

            # assign a soft value (]0, 1[) to the new voxels
            soft_new_voxels = lb * new_voxels

            # add the new voxels to the input mask
            arr_soft += soft_new_voxels
            arr_bin = (arr_soft > 0).astype(np.int)

        return arr_bin, arr_soft

    def dilate_arr(self, arr, dil_factor):
        # identify each object
        arr_labeled, lb_nb = label(arr.astype(np.int))

        # loop across each object
        arr_bin_lst, arr_soft_lst = [], []
        for obj_idx in range(1, lb_nb + 1):
            arr_bin_obj = (arr_labeled == obj_idx).astype(np.int)
            arr_soft_obj = np.copy(arr_bin_obj).astype(np.float)
            # compute the number of dilation iterations depending on the size of the lesion
            nb_it = int(round(dil_factor * math.sqrt(arr_bin_obj.sum())))
            # values of the voxels added to the input mask
            soft_label_values = [x / (nb_it + 1) for x in range(nb_it, 0, -1)]
            # dilate lesion
            arr_bin_dil, arr_soft_dil = self.dilate_lesion(arr_bin_obj, arr_soft_obj, soft_label_values)
            arr_bin_lst.append(arr_bin_dil)
            arr_soft_lst.append(arr_soft_dil)

        # sum dilated objects
        arr_bin_idx = np.sum(np.array(arr_bin_lst), axis=0)
        arr_soft_idx = np.sum(np.array(arr_soft_lst), axis=0)
        # clip values in case dilated voxels overlap
        arr_bin_clip, arr_soft_clip = np.clip(arr_bin_idx, 0, 1), np.clip(arr_soft_idx, 0.0, 1.0)

        return arr_soft_clip.astype(np.float), arr_bin_clip.astype(np.int)

    @staticmethod
    def random_holes(arr_in, arr_soft, arr_bin):
        arr_soft_out = np.copy(arr_soft)

        # coordinates of the new voxels, i.e. the ones from the dilation
        new_voxels_xx, new_voxels_yy, new_voxels_zz = np.where(np.logical_xor(arr_bin, arr_in))
        nb_new_voxels = new_voxels_xx.shape[0]

        # ratio of voxels added to the input mask from the dilated mask
        new_voxel_ratio = random.random()
        # randomly select new voxel indexes to remove
        idx_to_remove = random.sample(range(nb_new_voxels),
                                      int(round(nb_new_voxels * (1 - new_voxel_ratio))))

        # set to zero the here-above randomly selected new voxels
        arr_soft_out[new_voxels_xx[idx_to_remove],
                     new_voxels_yy[idx_to_remove],
                     new_voxels_zz[idx_to_remove]] = 0.0
        arr_bin_out = (arr_soft_out > 0).astype(np.int)

        return arr_soft_out, arr_bin_out

    @staticmethod
    def post_processing(arr_in, arr_soft, arr_bin, arr_dil):
        # remove new object that are not connected to the input mask
        arr_labeled, lb_nb = label(arr_bin)

        connected_to_in = arr_labeled * arr_in
        for lb in range(1, lb_nb + 1):
            if np.sum(connected_to_in == lb) == 0:
                arr_soft[arr_labeled == lb] = 0

        struct = np.ones((3, 3, 1) if arr_soft.shape[2] == 1 else (3, 3, 3))
        # binary closing
        arr_bin_closed = binary_closing((arr_soft > 0).astype(np.int), structure=struct)
        # fill binary holes
        arr_bin_filled = binary_fill_holes(arr_bin_closed)

        # recover the soft-value assigned to the filled-holes
        arr_soft_out = arr_bin_filled * arr_dil

        return arr_soft_out

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # binarize for processing
        gt_data_np = (sample > 0.5).astype(np.int_)

        if self.dil_factor > 0 and np.sum(sample):
            # dilation
            gt_dil, gt_dil_bin = self.dilate_arr(gt_data_np, self.dil_factor)

            # random holes in dilated area
            gt_holes, gt_holes_bin = self.random_holes(gt_data_np, gt_dil, gt_dil_bin)

            # post-processing
            gt_pp = self.post_processing(gt_data_np, gt_holes, gt_holes_bin, gt_dil)

            return gt_pp.astype(np.float32), metadata

        else:
            return sample, metadata


class RandomRotation(ImedTransform):
    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # If angle and metadata have been already defined for this sample, then use them
        if 'rotation' in metadata:
            angle, axes = metadata['rotation']
        # Otherwise, get random ones
        else:
            # Get the random angle
            angle = np.random.uniform(self.degrees[0], self.degrees[1])
            # Get the two axes that define the plane of rotation
            axes = tuple(random.sample(range(3 if sample.shape[2] > 1 else 2), 2))
            # Save params
            metadata['rotation'] = [angle, axes]

        # Do rotation
        data_out = rotate(sample,
                          angle=angle,
                          axes=axes,
                          reshape=False,
                          order=1).astype(sample.dtype)

        return data_out, metadata

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        # IMPORTANT NOTE: this function does not work with images (but works with labels)
        assert "rotation" in metadata
        # Opposite rotation, same axes
        angle, axes = - metadata['rotation'][0], metadata['rotation'][1]

        # Undo rotation
        data_out = rotate(sample,
                          angle=angle,
                          axes=axes,
                          reshape=False,
                          order=1).astype(sample.dtype)

        return data_out, metadata


class RandomReverse(ImedTransform):
    """Make a randomized symmetric inversion of the different values of each dimensions."""

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        if 'reverse' in metadata:
            flip_axes = metadata['reverse']
        else:
            # Flip axis booleans
            flip_axes = [np.random.randint(2) == 1 for _ in [0, 1, 2]]
            # Save in metadata
            metadata['reverse'] = flip_axes

        # Run flip
        for idx_axis, flip_bool in enumerate(flip_axes):
            if flip_axes:
                sample = np.flip(sample, axis=idx_axis).copy()

        return sample, metadata

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        assert "reverse" in metadata
        return self.__call__(sample, metadata)


class RandomAffine(RandomRotation):
    # TODO: implement scale and shear
    def __init__(self, degrees, translate=None):
        super().__init__(degrees)

        # Check Translate
        if translate is not None:
            assert isinstance(translate, (tuple, list)) and (len(translate) == 2 or len(translate) == 3), \
                "translate should be a list or tuple and it must be of length 2 or 3."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
            if len(translate) == 2:
                translate.append(0.0)
        self.translate = translate

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):

        # Get params
        if 'affine' in metadata:
            angle, axes_rot, translations = metadata['affine']
        else:
            self.data_shape = sample.shape

            angle = np.random.uniform(self.degrees[0], self.degrees[1])
            # Get the two axes that define the plane of rotation
            axes_rot = tuple(random.sample(range(3 if sample.shape[2] > 1 else 2), 2))

            if self.translate is not None:
                max_dx = self.translate[0] * self.data_shape[0]
                max_dy = self.translate[1] * self.data_shape[1]
                max_dz = self.translate[2] * self.data_shape[2]
                translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                                np.round(np.random.uniform(-max_dy, max_dy)),
                                np.round(np.random.uniform(-max_dz, max_dz)))
            else:
                translations = (0, 0, 0)

            metadata['affine'] = [angle, axes_rot, translations]

        # Run Rotation
        data_rot, _ = RandomRotation(self.degrees).__call__(sample, {'rotation': [angle, axes_rot]})
        # Run Translation
        data_rot_trans = shift(data_rot, shift=translations, order=1).astype(sample.dtype)
        # Run Scaling
        # data_rot_trans_scaled = zoom(data_rot_trans, zoom=scale, order=1)

        return data_rot_trans, metadata

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        # IMPORTANT NOTE: this function does not work with images (but works with labels)
        assert "affine" in metadata
        # Opposite rotation, same axesopposite translations
        angle, axes = - metadata['affine'][0], metadata['affine'][1]
        # Opposite translation
        translations = tuple([-t for t in metadata['affine'][2]])
        # Inverse scaling
        # scale = 1. / metadata['affine'][3]

        # Params
        dict_params = {"affine": [angle, axes, translations]}  # , scale]}

        # Undo rotation
        data_out, metadata = self.__call__(sample, dict_params)

        return data_out, metadata


class RandomShiftIntensity(ImedTransform):
    def __init__(self, shift_range, prob=0.1):
        self.shift_range = shift_range
        self.prob = prob

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        if np.random.random() < self.prob:
            # Get random offset
            offset = np.random.uniform(self.shift_range[0], self.shift_range[1])
        else:
            offset = 0.0

        # Update metadata
        metadata['offset'] = offset
        # Shift intensity
        data = (sample + offset).astype(sample.dtype)
        return data, metadata

    @multichannel_capable
    def undo_transform(self, sample, metadata=None):
        assert 'offset' in metadata
        # Get offset
        offset = metadata['offset']
        # Substract offset
        data = (sample - offset).astype(sample.dtype)
        return data, metadata


class ElasticTransform(ImedTransform):
    """Elastic transform for 2D and 3D inputs."""

    def __init__(self, alpha_range, sigma_range, p=0.1):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # if params already defined, i.e. sample is GT
        if "elastic" in metadata:
            alpha, sigma = metadata["elastic"]

        elif np.random.random() < self.p:
            # Get params
            alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            # Save params
            metadata["elastic"] = [alpha, sigma]

        else:
            metadata["elastic"] = [None, None]

        if any(metadata["elastic"]):
            # Get shape
            shape = sample.shape

            # Compute random deformation
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            dz = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            if shape[2] == 1:
                dz = 0  # No deformation along the last dimension
            x, y, z = np.meshgrid(np.arange(shape[0]),
                                  np.arange(shape[1]),
                                  np.arange(shape[2]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), \
                      np.reshape(y + dy, (-1, 1)), \
                      np.reshape(z + dz, (-1, 1))

            # Apply deformation
            data_out = map_coordinates(sample, indices, order=1, mode='reflect')
            # Keep input shape
            data_out = data_out.reshape(shape)
            # Keep data type
            data_out = data_out.astype(sample.dtype)

            return data_out, metadata

        else:
            return sample, metadata


class AdditiveGaussianNoise(ImedTransform):

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        if "gaussian_noise" in metadata:
            noise = metadata["gaussian_noise"]
        else:
            # Get random noise
            noise = np.random.normal(self.mean, self.std, sample.shape)
            noise = noise.astype(np.float32)

        # Apply noise
        data_out = sample + noise

        return data_out.astype(sample.dtype), metadata


class Clahe(ImedTransform):
    # TODO: Adapt to 3D
    def __init__(self, clip_limit=3.0, kernel_size=(8, 8)):
        # Default values are based upon the following paper:
        # https://arxiv.org/abs/1804.09400 (3D Consistent Cardiac Segmentation)
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        assert len(sample.shape) == 2
        assert len(self.kernel_size) == len(sample.shape)
        # Run equalization
        data_out = equalize_adapthist(sample,
                                      kernel_size=self.kernel_size,
                                      clip_limit=self.clip_limit).astype(sample.dtype)

        return data_out, metadata


class HistogramClipping(ImedTransform):

    def __init__(self, min_percentile=5.0, max_percentile=95.0):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        data = np.copy(sample)
        # Run clipping
        percentile1 = np.percentile(sample, self.min_percentile)
        percentile2 = np.percentile(sample, self.max_percentile)
        data[sample <= percentile1] = percentile1
        data[sample >= percentile2] = percentile2
        return data, metadata


def rescale_values_array(arr, minv=0.0, maxv=1.0, dtype=np.float32):
    """Rescale the values of numpy array `arr` to be from `minv` to `maxv`."""
    if dtype is not None:
        arr = arr.astype(dtype)

    mina = np.min(arr)
    maxa = np.max(arr)

    if mina == maxa:
        return arr * minv

    norm = (arr - mina) / (maxa - mina)  # normalize the array first
    return (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default

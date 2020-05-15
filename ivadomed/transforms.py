
import math
import numbers
import random
import functools
import numpy as np
from PIL import Image

from skimage.exposure import equalize_adapthist
from skimage.transform import resize
from scipy.ndimage import rotate, zoom
from scipy.ndimage.measurements import label, center_of_mass
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes, binary_closing
from scipy.ndimage.interpolation import map_coordinates

import torch
import torch.nn.functional as F
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

    def __call__(self, sample, metadata={}):
        raise NotImplementedError("You need to implement the transform() method.")

    def undo_transform(self, sample, metadata={}):
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
            transform_obj = globals()[transform](**parameters)

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
        if self.transform[data_type] is None:
            # In case self.transform[data_type] is None
            return None, None
        else:
            for tr in list(self.transform[data_type]):
                sample, metadata = tr(sample, metadata)
            return sample, metadata


class UndoCompose(object):
    def __init__(self, compose):
        self.transforms = compose.transforms

    def __call__(self, img):
        for t in self.transforms[::-1]:
            img = t.undo_transform(img)
        return img


class UndoTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform.undo_transform(sample)


class NumpyToTensor(ImedTransform):
    """Converts numpy array to tensor object."""

    @multichannel_capable
    def undo_transform(self, sample, metadata={}):
        return sample.numpy(), metadata

    @multichannel_capable
    def __call__(self, sample, metadata={}):
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

    def __init__(self, hspace, wspace, dspace=1., interpolation_order=2):
        self.hspace = hspace
        self.wspace = wspace
        self.dspace = dspace
        self.interpolation_order = interpolation_order

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata):
        assert "resample" in metadata

        # Get params
        params_do = metadata["resample"]
        params_undo = [1 / x for x in params_do]

        # Undo resampling
        data_out = zoom(sample,
                        zoom=params_undo,
                        order=self.interpolation_order)

        # Data type
        data_out = data_out.astype(sample.dtype)

        return data_out, metadata

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata):
        # Get params
        # Voxel dimension in mm
        zooms = metadata["zooms"]
        if len(zooms) == 2:
            zooms += tuple([1.0])
        hfactor = zooms[0] / self.hspace
        wfactor = zooms[1] / self.wspace
        dfactor = zooms[2] / self.dspace
        params_resample = (hfactor, wfactor, dfactor)

        # Save params
        metadata['resample'] = params_resample

        # Run resampling
        data_out = zoom(sample,
                        zoom=params_resample,
                        order=self.interpolation_order)

        # Data type
        data_out = data_out.astype(sample.dtype)

        return data_out, metadata


# TODO
class Normalize(ImedTransform):
    """Normalize a tensor image with mean and standard deviation.
    :param mean: mean value.
    :param std: standard deviation value.
    In case of multiple inputs, both mean and std are lists.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        input_data = sample['input']
        # TODO: Decorator?
        # Normalize
        if isinstance(input_data, list):
            # TODO: .instance_norm?
            input_data = [F.normalize(input_data[i], self.mean[i], self.std[i]) for i in range(len(input_data))]
        else:
            # TODO: .instance_norm?
            input_data = F.normalize(input_data, self.mean, self.std)

        # Update
        rdict = {'input': input_data}
        sample.update(rdict)
        return sample


class NormalizeInstance(ImedTransform):
    """Normalize a tensor or an array image with mean and standard deviation estimated
    from the sample itself.
    """

    @multichannel_capable
    def __call__(self, sample, metadata={}):
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
                newitem = [slice(None)]*self.ndim
                newitem[0] = item
                item = newitem
            # We're out of items, just append noop slices
            if dim >= len(item):
                all_in_slices.append(slice(0, self.shape[dim]))
                pad.append((0, 0))
            # We're dealing with an integer (no padding even if it's
            # out of bounds)
            if isinstance(item[dim], int):
                all_in_slices.append(slice(item[dim], item[dim]+1))
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
        self.is_2D = True if len(size) == 2 else False

    @staticmethod
    def _adjust_padding(npad, sample):
        npad_out_tuple = []
        for idx_dim, tuple_pad in enumerate(npad):
            pad_start, pad_end = tuple_pad
            if pad_start < 0 or pad_end < 0:
                # Move axis of interest
                sample_reorient = np.swapaxes(sample, 0, idx_dim)
                # Adjust pad and crop
                if pad_start < 0:
                    sample_crop = sample_reorient[abs(pad_start):, :]
                    pad_start = 0
                if pad_end < 0:
                    sample_crop = sample_reorient[:pad_end, :]
                    pad_end = 0
                # Reorient
                sample = np.swapaxes(sample_crop, 0, idx_dim)

            npad_out_tuple.append((pad_start, pad_end))

        return npad_out_tuple, sample

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata={}):
        # Get params
        th, tw, td = self.size
        fh, fw, fd, h, w, d = metadata['crop_params']

        # Crop data
        # Note we use here CroppableArray in order to deal with "out of boundaries" crop
        # e.g. if fh is negative or fh+th out of bounds, then it will pad
        if self.is_2D:
            data_out = sample.view(CroppableArray)[fh:fh + th, fw:fw + tw, :]
        else:
            data_out = sample.view(CroppableArray)[fh:fh+th, fw:fw+tw, fd:fd+td]

        return data_out, metadata

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata):
        # Get crop params
        th, tw, td = self.size
        fh, fw, fd, h, w, d = metadata["crop_params"]

        # Compute params to undo transform
        pad_left = fw
        pad_right = w - pad_left - tw
        pad_top = fh
        pad_bottom = h - pad_top - th
        pad_front = fd
        pad_back = d - pad_front - td if not self.is_2D else 0
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
    def __call__(self, sample, metadata={}):
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
    def __call__(self, sample, metadata={}):
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


# TODO
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
        new_voxels_xx, new_voxels_yy = np.where(np.logical_xor(arr_bin, arr_in))
        nb_new_voxels = new_voxels_xx.shape[0]

        # ratio of voxels added to the input mask from the dilated mask
        new_voxel_ratio = random.random()
        # randomly select new voxel indexes to remove
        idx_to_remove = random.sample(range(nb_new_voxels),
                                      int(round(nb_new_voxels * (1 - new_voxel_ratio))))

        # set to zero the here-above randomly selected new voxels
        arr_soft_out[new_voxels_xx[idx_to_remove], new_voxels_yy[idx_to_remove]] = 0.0
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

        # binary closing
        arr_bin_closed = binary_closing((arr_soft > 0).astype(bool))
        # fill binary holes
        arr_bin_filled = binary_fill_holes(arr_bin_closed)

        # recover the soft-value assigned to the filled-holes
        arr_soft_out = arr_bin_filled * arr_dil

        return arr_soft_out

    def __call__(self, sample):
        gt_data = sample['gt']
        gt_t = []
        for gt in gt_data:
            gt_data_np = np.array(gt)
            # binarize for processing
            gt_data_np = (gt_data_np > 0.5).astype(np.int_)

            if self.dil_factor > 0 and np.sum(gt):
                # dilation
                gt_dil, gt_dil_bin = self.dilate_arr(gt_data_np, self.dil_factor)

                # random holes in dilated area
                gt_holes, gt_holes_bin = self.random_holes(gt_data_np, gt_dil, gt_dil_bin)

                # post-processing
                gt_pp = self.post_processing(gt_data_np, gt_holes, gt_holes_bin, gt_dil)

                # mask with ROI
                if sample['roi'][0] is not None:
                    gt_pp[np.array(sample['roi'][0]) == 0] = 0.0

                gt_t.append(Image.fromarray(gt_pp))

        if len(gt_t):
            rdict = {'gt': gt_t}
            sample.update(rdict)

        return sample


# TODO: unit test
class AddBackgroundClass(ImedTransform):

    # TODO
    #def undo_transform(self, sample):
    #    return sample

    # Note: We do not apply @multichannel_capable to AddBackgroundClass
    # because we need all the channels to determine the Background.
    def __call__(self, sample, metadata={}):
        # Sum across the channels (i.e. labels)
        sum_labels = np.sum(sample, axis=0)
        # Get background
        background = (sum_labels == 0).astype(sample.dtype)
        # Expand dim
        background = np.expand_dims(background, axis=0)
        # Concatenate
        data_out = np.concatenate(background, sample)

        return data_out, metadata


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
    def __call__(self, sample, metadata={}):
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
    def undo_transform(self, sample, metadata):
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


# TODO
class RandomReverse3D(ImedTransform):
    """Make a randomized symmetric inversion of the different values of each dimensions."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}

        input_list = sample['input']

        # TODO: Generalize this in constructor?
        if not isinstance(input_list, list):
            input_list = [sample['input']]

        # Flip axis booleans
        flip_axes = [np.random.randint(2) == 1 for axis in [0, 1, 2]]

        # Run flip
        reverse_input = []
        for input_data in input_list:
            for idx_axis, flip_bool in enumerate(flip_axes):
                if flip_axes:
                    input_data = np.flip(input_data, axis=idx_axis).copy()
            reverse_input.append(input_data)

        # Update
        rdict['input'] = reverse_input

        # Labeled data
        if self.labeled:
            gt_data = sample['gt']
            for idx_axis, flip_bool in enumerate(flip_axes):
                if flip_axes:
                    gt_data = np.flip(gt_data, axis=idx_axis).copy()
            rdict['gt'] = gt_data

        # Update
        sample.update(rdict)
        return sample


# TODO
class RandomAffine(RandomRotation):

    def __init__(self, degrees, translate=None,
                 scale=None, shear=None,
                 resample=False, fillcolor=0):
        super().__init__(degrees)

        # Check Translate
        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        # Check scale
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        # Check shear
        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    def get_params(self):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.translate is not None:
            max_dx = self.translate[0] * self.input_data_size[0]
            max_dy = self.translate[1] * self.input_data_size[1]
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if self.scale is not None:
            scale = np.random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.0

        if self.shear is not None:
            shear = np.random.uniform(self.shear[0], self.shear[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def sample_augment(self, input_data, params):
        input_data = F.affine(input_data, *params, resample=self.resample,
                              fillcolor=self.fillcolor)
        return input_data

    def label_augment(self, gt_data, params):
        gt_data = self.sample_augment(gt_data, params)
        np_gt_data = np.array(gt_data)
        np_gt_data[np_gt_data >= 0.5] = 255.0
        np_gt_data[np_gt_data < 0.5] = 0.0
        np_gt_data = np_gt_data.astype(np.uint8)
        gt_data = Image.fromarray(np_gt_data, mode='L')
        return gt_data

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        rdict = {}
        input_data = sample['input']

        self.input_data_size = input_data[0].size

        params = self.get_params()

        if isinstance(input_data, list):
            ret_input = [self.sample_augment(item, params)
                         for item in input_data]
        else:
            ret_input = self.sample_augment(input_data, params)
        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            if isinstance(gt_data, list):
                ret_gt = [self.label_augment(item, params)
                          for item in gt_data]
            else:
                ret_gt = self.label_augment(gt_data, params)
            rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample


# TODO
class RandomAffine3D(RandomAffine):

    def __call__(self, sample):
        rdict = {}

        input_data = sample['input']
        # TODO: To generalize?
        if not isinstance(input_data, list):
            input_data = [sample['input']]

        self.input_data_size = input_data[0][0, :, :].shape
        params = self.get_params()

        ret_input = []
        for volume in input_data:
            img_data = np.zeros(input_data[0].shape)
            for idx, img in enumerate(volume):
                pil_img = Image.fromarray(img, mode='F')
                img_data[idx, :, :] = np.array(self.sample_augment(pil_img, params))
            ret_input.append(img_data.astype('float32'))

        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            ret_gt = []
            for labels in gt_data:
                gt_vol = np.zeros(labels.shape)
                for idx, gt in enumerate(labels):
                    pil_img = Image.fromarray(gt, mode='F')
                    gt_vol[idx, :, :] = np.array(self.sample_augment(pil_img, params))
                ret_gt.append(gt_vol.astype('float32'))
            rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample


class RandomShiftIntensity(ImedTransform):

    def __init__(self, shift_range, prob=0.1):
        self.shift_range = shift_range
        self.prob = prob

    @multichannel_capable
    def __call__(self, sample, metadata={}):
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
    def undo_transform(self, sample, metadata={}):
        assert 'offset' in metadata
        # Get offset
        offset = metadata['offset']
        # Substract offset
        data = (sample - offset).astype(sample.dtype)
        return data, metadata


class ElasticTransform(ImedTransform):
    """Elastic transform for 2D and 3D inputs."""

    def __init__(self, alpha_range, sigma_range):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata={}):
        # if params already defined, i.e. sample is GT
        if "elastic" in metadata:
            alpha, sigma = metadata["elastic"]
        else:
            # Get params
            alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            # Save params
            metadata["elastic"] = [alpha, sigma]

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
        indices = np.reshape(x + dx, (-1, 1)),\
                  np.reshape(y + dy, (-1, 1)),\
                  np.reshape(z + dz, (-1, 1))

        # Apply deformation
        data_out = map_coordinates(sample, indices, order=1, mode='reflect')
        # Keep input shape
        data_out = data_out.reshape(shape)
        # Keep data type
        data_out = data_out.astype(sample.dtype)

        return data_out, metadata


# TODO
class AdditiveGaussianNoise(ImedTransform):

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        # Get random noise
        noise = np.random.normal(self.mean, self.std, input_data[0].size)
        noise = noise.astype(np.float32)

        # Apply noise
        noisy_input = []
        for item in input_data:
            np_input_data = np.array(item)
            np_input_data += noise
            noisy_input.append(Image.fromarray(np_input_data, mode='F'))

        # Update
        rdict['input'] = noisy_input
        sample.update(rdict)
        return sample


# TODO
class Clahe(ImedTransform):

    def __init__(self, clip_limit=3.0, kernel_size=(8, 8)):
        # Default values are based upon the following paper:
        # https://arxiv.org/abs/1804.09400 (3D Consistent Cardiac Segmentation)
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    def do_clahe(self, data):
        data = np.copy(data)
        # Ensure that data is a numpy array
        data = np.array(data)
        # Run equalization
        clahe_data = equalize_adapthist(data,
                                        kernel_size=self.kernel_size,
                                        clip_limit=self.clip_limit)
        return clahe_data

    def __call__(self, sample):
        input_data = sample['input']

        # TODO: Decorator?
        if isinstance(input_data, list):
            output_data = [self.do_clahe(data) for data in input_data]
        else:
            output_data = self.do_clahe(input_data)

        # Update
        rdict = {'input': output_data}
        sample.update(rdict)
        return sample


class HistogramClipping(ImedTransform):

    def __init__(self, min_percentile=5.0, max_percentile=95.0):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    @multichannel_capable
    def __call__(self, sample, metadata={}):
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

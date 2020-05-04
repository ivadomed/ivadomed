import math
import numbers
import random
import numpy as np
from PIL import Image

from skimage.exposure import equalize_adapthist
from scipy.ndimage.measurements import label, center_of_mass
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes, binary_closing
from scipy.ndimage.interpolation import map_coordinates

import torch
import torchvision.transforms.functional as F
from torchvision import transforms as torchvision_transforms


class IMEDTransform(object):

    def __call__(self, sample):
        raise NotImplementedError("You need to implement the transform() method.")

    def undo_transform(self, sample):
        raise NotImplementedError("You need to implement the undo_transform() method.")


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


class ToPIL(IMEDTransform):
    """Converts array or tensor to PIL object."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    @staticmethod
    def sample_transform(sample_data):
        # Numpy array
        if not isinstance(sample_data, np.ndarray):
            input_data_npy = sample_data.numpy()
        else:
            input_data_npy = sample_data

        input_data = Image.fromarray(input_data_npy, mode='F')
        return input_data

    def __call__(self, sample):
        rdict = {}

        # Input data
        input_data = sample['input']
        ret_input = [self.sample_transform(item) for item in input_data]
        rdict['input'] = ret_input

        # Labeled data
        if self.labeled:
            gt_data = sample['gt']
            ret_gt = [self.sample_transform(item) for item in gt_data]
            rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample


def compose_transforms(dict_transforms, requires_undo=False):
    """Composes several transforms together.
    Args:
        dict_transforms (dictionary): Dictionary where the keys are the transform names
            and the value their parameters.
        requires_undo (bool): If True, does not include transforms which do not have an undo_transform
            implemented yet.
    Returns:
        torchvision.transforms.Compose object.
    """
    list_transform = []
    for transform in dict_transforms.keys():
        parameters = dict_transforms[transform]

        # call transfrom
        transform_obj = globals()[transform](**parameters)

        # check if undo_transform method is implemented
        if requires_undo:
            if hasattr(transform_obj, 'undo_transform'):
                list_transform.append(transform_obj)
            else:
                print('{} transform not included since no undo_transform available for it.'.format(transform))
        else:
            list_transform.append(transform_obj)

    return torchvision_transforms.Compose(list_transform)


class Resample(IMEDTransform):

    def __init__(self, wspace, hspace,
                 interpolation=Image.BILINEAR,
                 labeled=True):
        self.hspace = hspace
        self.wspace = wspace
        self.interpolation = interpolation
        self.labeled = labeled

    @staticmethod
    def do_resample(list_data, new_shape, interpolation_mode):
        list_data_out = []
        for i, data in enumerate(list_data):
            resampled_data = data.resize(new_shape,
                                         resample=interpolation_mode)
            list_data_out.append(resampled_data)
        return list_data_out

    def undo_transform(self, sample):
        rdict = {}

        # Get original data shape
        hshape, wshape = sample['input_metadata']['data_shape']

        # Input data
        rdict['input'] = self.do_resample(list_data=sample["input"],
                                          new_shape=(wshape, hshape),
                                          interpolation_mode=self.interpolation)

        # Prediction data
        # Note: We use here self.interpolation instead of forcing Image.NEAREST
        #       in order to ensure soft output
        rdict['gt'] = self.do_resample(list_data=sample["gt"],
                                       new_shape=(wshape, hshape),
                                       interpolation_mode=self.interpolation)

        # Update
        sample.update(rdict)
        return sample

    def __call__(self, sample):
        rdict = {}

        # Get new data shape
        # Based on the assumption that the metadata of every modality are equal.
        # Voxel dimension in mm
        input_metadata = sample['input_metadata']
        hzoom, wzoom = input_metadata[0]["zooms"]
        hshape, wshape = input_metadata[0]["data_shape"]
        hfactor = hzoom / self.hspace
        wfactor = wzoom / self.wspace
        hshape_new = int(round(hshape * hfactor))
        wshape_new = int(round(wshape * wfactor))

        # Input data
        rdict['input'] = self.do_resample(list_data=sample["input"],
                                          new_shape=(wshape_new, hshape_new),
                                          interpolation_mode=self.interpolation)

        # Labeled data
        if self.labeled:
            rdict['gt'] = self.do_resample(list_data=sample["gt"],
                                           new_shape=(wshape_new, hshape_new),
                                           interpolation_mode=Image.NEAREST)

        # ROI data
        if sample['roi'] is not None:
            rdict['roi'] = self.do_resample(list_data=sample["roi"],
                                            new_shape=(wshape_new, hshape_new),
                                            interpolation_mode=Image.NEAREST)

        # Update
        sample.update(rdict)
        return sample


class Normalize(IMEDTransform):
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


class NormalizeInstance(IMEDTransform):
    """Normalize a tensor image with mean and standard deviation estimated
    from the sample itself.
    """

    @staticmethod
    def do_normalize(data):
        # TODO: instance_norm?
        if data.type(torch.bool).any():
            mean, std = data.mean(), data.std()
            return F.normalize(data, [mean], [std])
        else:
            return data

    def __call__(self, sample):
        input_data = sample['input']

        # TODO: Decorator?
        # Normalize
        if isinstance(input_data, list):
            input_data_normalized = []
            for i in range(len(input_data)):
                input_data_normalized.append(self.do_normalize(input_data[i]))
        else:
            input_data_normalized = self.do_normalize(input_data)

        # Update
        rdict = {'input': input_data_normalized}
        sample.update(rdict)
        return sample

    def undo_transform(self, sample):
        return sample


class ToTensor(IMEDTransform):
    """Convert a PIL image(s) or numpy array(s) to a PyTorch tensor(s)."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        # TODO: function?
        # Input data
        if len(input_data) > 1:
            # Multiple inputs
            ret_input = [F.to_tensor(item) for item in input_data]
        else:
            # single input
            ret_input = F.to_tensor(input_data[0])
        rdict['input'] = ret_input

        # Labeled data
        if self.labeled:
            gt_data = sample['gt']
            if gt_data is not None:
                if isinstance(gt_data, list):
                    # multiple GT
                    # torch.cat is used to be compatible with StackTensors
                    ret_gt = torch.cat([F.to_tensor(item) for item in gt_data], dim=0)

                    # Add dim 0 for 3D images (i.e. 2D slices with multiple GT)
                    if isinstance(gt_data[0], np.ndarray) and len(gt_data[0].shape) == 3:
                        ret_gt = ret_gt.unsqueeze(0)

                else:
                    # single GT
                    ret_gt = F.to_tensor(gt_data)

                rdict['gt'] = ret_gt

        # Update sample
        sample.update(rdict)
        return sample

    def undo_transform(self, sample):
        # Returns a PIL object
        return ToPIL()(sample)


class Crop2D(IMEDTransform):

    def __init__(self, size, labeled=True):
        self.size = size
        self.labeled = labeled

    # TODO: return
    @staticmethod
    def propagate_params(sample, params, i):
        input_metadata = sample['input_metadata'][i]
        input_metadata["__centercrop"] = params
        return input_metadata

    @staticmethod
    def get_params(sample):
        return [sample['input_metadata'][i]["__centercrop"] for i in range(len(sample))]


class CenterCrop2D(Crop2D):
    """Make a centered crop of a specified size."""

    def __init__(self, size, labeled=True):
        super().__init__(size, labeled)

    def __call__(self, sample):
        rdict = {}
        th, tw = self.size

        # Input data
        input_data = sample['input']

        # As the modalities are registered, the cropping params are the same
        w, h = input_data[0].size
        fh = int(round((h - th) / 2.))
        fw = int(round((w - tw) / 2.))
        params = (fh, fw, w, h)

        # Loop across input modalities
        for i in range(len(input_data)):
            # Updating the parameters in the input metadata
            self.propagate_params(sample, params, i)
            # Cropping
            input_data[i] = F.center_crop(input_data[i], self.size)

        # Update
        rdict['input'] = input_data

        # Labeled data
        if self.labeled:
            gt_data = sample['gt']
            # Loop across GT
            for i in range(len(gt_data)):
                # Cropping
                gt_data[i] = F.center_crop(gt_data[i], self.size)

            # Update
            rdict['gt'] = gt_data

        sample.update(rdict)
        return sample

    def do_uncrop(self, data, params):
        fh, fw, w, h = params
        th, tw = self.size
        pad_left = fw
        pad_right = w - pad_left - tw
        pad_top = fh
        pad_bottom = h - pad_top - th
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return F.pad(data, padding)

    def undo_transform(self, sample):
        rdict = {}

        crop_params = sample['input_metadata']["__centercrop"]

        # TODO: Decorator?
        # Input data
        if isinstance(sample['input'], list):
            rdict['input'] = sample['input']
            for i in range(len(sample['input'])):
                # TODO: sample['input_metadata'][i]["__centercrop"]
                rdict['input'][i] = self.do_uncrop(data=sample['input'][i],
                                                   params=crop_params)
        else:
            rdict['input'] = self.do_uncrop(data=sample['input'],
                                            params=crop_params)

        # Labeled data
        # Note: undo_transform: we force labeled=True because used with predictions
        rdict['gt'] = sample['gt']
        for i in range(len(sample['gt'])):
            rdict['gt'][i] = self.do_uncrop(data=sample['gt'][i],
                                            params=crop_params)

        # Update
        sample.update(rdict)
        return sample


class ROICrop2D(Crop2D):
    """Make a crop of a specified size around a ROI."""

    def __init__(self, size, labeled=True):
        super().__init__(size, labeled)

    def do_uncrop(self, data, params):
        fh, fw, w, h = params
        tw, th = self.size
        pad_left = fw
        pad_right = w - pad_left - tw
        pad_top = fh
        pad_bottom = h - pad_top - th
        padding = (pad_top, pad_left, pad_bottom, pad_right)
        return F.pad(data, padding)

    def undo_transform(self, sample):
        rdict = {
            'input': [],
            'gt': []
        }

        crop_params = sample['input_metadata']["__centercrop"]

        # TODO: call get_params
        # TODO: Decorator?
        if isinstance(sample['input'], list):
            for input_data in sample['input']:
                rdict['input'].append(self.do_uncrop(data=input_data,
                                                     params=crop_params))
        else:
            rdict['input'] = self.do_uncrop(data=sample['input'],
                                            params=crop_params)

        # Note: undo_transform: we force labeled=True because used with predictions
        for gt in sample['gt']:
            rdict['gt'].append(self.do_uncrop(data=gt,
                                              params=crop_params))

        sample.update(rdict)
        return sample

    def __call__(self, sample):
        rdict = {}

        input_data = sample['input']
        roi_data = sample['roi'][0]

        # compute center of mass of the ROI
        x_roi, y_roi = center_of_mass(np.array(roi_data).astype(np.int))
        x_roi, y_roi = int(round(x_roi)), int(round(y_roi))
        tw, th = self.size
        th_half, tw_half = int(round(th / 2.)), int(round(tw / 2.))

        # compute top left corner of the crop area
        fh = y_roi - th_half
        fw = x_roi - tw_half
        # params are shared across modalities because are all registered
        w, h = input_data[0].size
        params = (fh, fw, h, w)
        self.propagate_params(sample, params, 0)

        # Loop across modalities
        for i in range(len(input_data)):
            # crop data
            input_data[i] = F.crop(input_data[i], fw, fh, tw, th)
        # Update
        rdict['input'] = input_data

        # Labeled data
        if self.labeled:
            gt_data = sample['gt']
            for i in range(len(gt_data)):
                gt_data[i] = F.crop(gt_data[i], fw, fh, tw, th)
            rdict['gt'] = gt_data

        sample.update(rdict)
        return sample


class DilateGT(IMEDTransform):
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


class StackTensors(IMEDTransform):
    """Stack all modalities, as a list, in a single tensor."""

    def __call__(self, sample):
        rdict = {}

        # Input data
        if isinstance(sample['input'], list):
            input_data = sample['input']
            rdict['input'] = torch.cat(input_data, dim=0)
            sample.update(rdict)

        # Labeled data
        if isinstance(sample['gt'], list):
            gt_data = sample['gt']
            rdict['gt'] = torch.cat(gt_data, dim=0)
            sample.update(rdict)

        return sample

    def undo_transform(self, sample):
        return sample


class CenterCrop3D(IMEDTransform):
    """Make a centered crop of a specified size.
    :param labeled: if it is a segmentation task.
                         When this is True (default), the crop
                         will also be applied to the ground truth.
    """

    def __init__(self, size, labeled=True):
        self.size = size
        self.labeled = labeled

    def undo_transform(self, sample):
        # TODO: Make it compatible with lists
        # Compute parameters
        td, tw, th = sample['input_metadata']["__centercrop"]
        d, w, h = sample['input_metadata']["data_shape"]
        fh = max(int(round((h - th) / 2.)), 0)
        fw = max(int(round((w - tw) / 2.)), 0)
        fd = max(int(round((d - td) / 2.)), 0)
        npad = ((0, 0), (fw, fw), (fd, fd), (fh, fh))

        # Apply undo
        input_undo = np.pad(sample['input'],
                            pad_width=npad,
                            mode='constant',
                            constant_values=0)
        gt_undo = np.pad(sample['input'],
                         pad_width=npad,
                         mode='constant',
                         constant_values=0)

        # Update
        rdict = {'input': input_undo, 'gt': gt_undo}
        sample.update(rdict)
        return sample

    def do_crop(self, list_data, crop_params, cval=None):
        td, tw, th = self.size
        fd, fw, fh = crop_params

        list_crop_data = []
        for data in list_data:
            # Do crop
            crop_data = data[fd:fd + td, fw:fw + tw, fh:fh + th]

            # Pad image with mean if image smaller than crop size
            cd, cw, ch = crop_data.shape
            if (cw, ch, cd) != (tw, th, td):
                w_diff = (tw - cw) / 2.
                iw = 1 if w_diff % 1 != 0 else 0
                h_diff = (th - ch) / 2.
                ih = 1 if h_diff % 1 != 0 else 0
                d_diff = (td - cd) / 2.
                id_ = 1 if d_diff % 1 != 0 else 0
                npad = ((int(d_diff) + id_, int(d_diff)),
                        (int(w_diff) + iw, int(w_diff)),
                        (int(h_diff) + ih, int(h_diff)))
                constant_values = cval if not cval is None else np.mean(crop_data)
                crop_data = np.pad(crop_data,
                                   pad_width=npad,
                                   mode='constant',
                                   constant_values=constant_values)

            list_crop_data.append(crop_data)

        return list_crop_data

    def __call__(self, sample):
        # TODO: compatible with non list
        # TODO: ROI

        # Get params
        d, w, h = sample['input'][0].shape
        td, tw, th = self.size
        fh = max(int(round((h - th) / 2.)), 0)
        fw = max(int(round((w - tw) / 2.)), 0)
        fd = max(int(round((d - td) / 2.)), 0)
        crop_params = (fd, fw, fh)

        # Propagate params
        for idx, img in enumerate(sample['input']):
            sample['input_metadata'][idx]["__centercrop"] = td, tw, th

        # Do crop
        do_input = self.do_crop(list_data=sample['input'], crop_params=crop_params)
        rdict = {'input': do_input,
                 'input_metadata': sample['input_metadata']}

        if self.labeled:
            do_gt = self.do_crop(list_data=sample['gt'], crop_params=crop_params, cval=0)
            rdict['gt'] = do_gt

        # Update
        sample.update(rdict)
        return sample


class NormalizeInstance3D(IMEDTransform):

    @staticmethod
    def do_normalize(data):
        # TODO: instance_norm?
        # Check if not empty
        if data.type(torch.bool).any():
            mean, std = data.mean(), data.std()
            return F.normalize(data,
                               [mean for _ in range(0, data.shape[0])],
                               [std for _ in range(0, data.shape[0])]).unsqueeze(0)
        else:
            return data

    def __call__(self, sample):
        input_data = sample['input']

        # TODO: Decorator
        if isinstance(input_data, list):
            input_data_normalized = []
            for i in range(len(input_data)):
                input_data_normalized.append(self.do_normalize(input_data[i]))
        else:
            input_data_normalized = self.do_normalize(input_data)

        # Update
        rdict = {'input': input_data_normalized}
        sample.update(rdict)
        return sample

    def undo_transform(self, sample):
        return sample


class BackgroundClass(IMEDTransform):

    def undo_transform(self, sample):
        return sample

    def __call__(self, sample):
        rdict = {}

        background = (sample['gt'].sum(axis=0) == 0).type('torch.FloatTensor')[None,]
        rdict['gt'] = torch.cat((background, sample['gt']), dim=0)

        sample.update(rdict)
        return sample


class RandomRotation(IMEDTransform):
    def __init__(self, degrees, resample=False,
                 expand=False, center=None,
                 labeled=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.labeled = labeled

    def get_params(self):
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        return angle

    def do_rotate(self, data, angle):
        return F.rotate(data, angle, self.resample, self.expand, self.center)

    def __call__(self, sample):
        rdict = {}

        input_data = sample['input']

        angle = self.get_params()

        # save angle in metadata
        rdict['input_metadata'] = sample['input_metadata']

        # Input data
        input_transform = []
        # TODO: Always a list?
        # TODO: Decorator or function?
        for idx, input_data in enumerate(input_data):
            rdict['input_metadata'][idx]['randomRotation'] = angle
            input_transform.append(self.do_rotate(input_data, angle))
        rdict['input'] = input_transform

        # Labeled data
        if self.labeled:
            gt_transform = []
            for gt_data in sample['gt']:
                gt_transform.append(self.do_rotate(gt_data, angle))
            rdict['gt'] = gt_transform

        # Update
        sample.update(rdict)
        return sample

    def undo_transform(self, sample):
        rdict = {}
        # Opposite rotation
        angle = - sample['input_metadata']['randomRotation']

        # TODO: Decorator or function?
        if isinstance(sample['input'], list):
            rdict['input'] = sample['input']
            for i in range(len(sample['input'])):
                rdict['input'][i] = self.do_rotate(sample['input'][i], angle)
        else:
            rdict['input'] = self.do_rotate(sample['input'], angle)

        if isinstance(sample['gt'], list):
            rdict['gt'] = sample['gt']
            for i in range(len(sample['gt'])):
                rdict['gt'][i] = self.do_rotate(sample['gt'][i], angle)
        else:
            rdict['gt'] = self.do_rotate(sample['gt'], angle)

        # Update
        sample.update(rdict)
        return sample


# TODO: Merge RandomRotation and RandomRotation3D
class RandomRotation3D(RandomRotation):
    """Make a rotation of the volume's values.

    :param degrees: Maximum rotation's degrees.
    :param axis: Axis of the rotation.
    :param labeled: Boolean if label data is provided.
    """

    def __init__(self, degrees, axis=0, labeled=True):
        super().__init__(degrees, labeled=labeled)
        self.axis = axis

    # TODO: use code of RandomRotation that could output a list then reconstruct 3D arr
    def do_rotate(self, data, angle):
        for x in range(data.shape[0]):
            data[x, :, :] = F.rotate(Image.fromarray(data[x, :, :], mode='F'), angle)
        return data

    def __call__(self, sample):
        rdict = {}

        input_data = sample['input']

        # Check if input data is 3D
        if len(sample['input'][0].shape) != 3:
            raise ValueError("Input of RandomRotation3D should be a 3 dimensionnal tensor.")

        # Get angles
        angle = self.get_params()

        # Move axis of interest to the first dimension
        input_data_movedaxis = np.moveaxis(input_data, self.axis, 0)

        # Transform data
        input_rotated = self.do_rotate(input_data_movedaxis, angle)

        # Reorient data
        input_out = np.moveaxis(input_rotated, 0, self.axis)

        # Update
        rdict['input'] = input_out

        # If labeled data
        if self.labeled:
            gt_data = sample['gt']
            gt_data_movedaxis = np.moveaxis(gt_data, self.axis, 0)
            gt_rotated = self.do_rotate(gt_data_movedaxis, angle)
            gt_out = np.moveaxis(gt_rotated, 0, self.axis)
            rdict['gt'] = gt_out

        sample.update(rdict)
        return sample


# TODO: Extend to 2D?
class RandomReverse3D(IMEDTransform):
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


class RandomAffine(RandomRotation):

    def __init__(self, degrees, translate=None,
                 scale=None, shear=None,
                 resample=False, fillcolor=0,
                 labeled=True):
        super().__init__(degrees, labeled=labeled)

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


class RandomTensorChannelShift(IMEDTransform):

    def __init__(self, shift_range):
        self.shift_range = shift_range

    @staticmethod
    def get_params(shift_range):
        sampled_value = np.random.uniform(shift_range[0],
                                          shift_range[1])
        return sampled_value

    @staticmethod
    def sample_augment(input_data, params):
        np_input_data = np.array(input_data)
        np_input_data += params
        input_data = Image.fromarray(np_input_data, mode='F')
        return input_data

    def __call__(self, sample):
        input_data = sample['input']
        params = self.get_params(self.shift_range)

        if isinstance(input_data, list):
            ret_input = [self.sample_augment(item, params) for item in input_data]
        else:
            ret_input = self.sample_augment(input_data, params)

        rdict = {'input': ret_input}

        sample.update(rdict)
        return sample


class ElasticTransform(IMEDTransform):
    "Elastic transform for 2D and 3D inputs"

    def __init__(self, alpha_range, sigma_range,
                 p=0.5, labeled=True):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.labeled = labeled
        self.p = p
        self.is3D = False  # Set as default, it will be re-assesed when called

    @staticmethod
    def get_params(alpha, sigma):
        alpha = np.random.uniform(alpha[0], alpha[1])
        sigma = np.random.uniform(sigma[0], sigma[1])
        return alpha, sigma

    def elastic_transform(self, image, alpha, sigma):
        shape = image.shape

        # Compute random deformation
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha

        if self.is3D:
            dz = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            x, y, z = np.meshgrid(np.arange(shape[0]),
                                  np.arange(shape[1]),
                                  np.arange(shape[2]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))
        else:
            x, y = np.meshgrid(np.arange(shape[0]),
                               np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        # Apply deformation
        return map_coordinates(image, indices, order=1).reshape(shape)

    def sample_augment(self, input_data, params):
        param_alpha, param_sigma = params

        np_input_data = np.array(input_data)

        # Check if input is 3D data
        if len(np_input_data.shape) == 3:
            self.is3D = True

        # Run transform
        np_output_data = self.elastic_transform(np_input_data,
                                                param_alpha, param_sigma)

        # TODO: CHeck with Andreanne
        if self.is3D:
            output_data = np_output_data
        else:
            output_data = Image.fromarray(np_output_data, mode='F')

        return output_data

    def label_augment(self, gt_data, params):
        param_alpha, param_sigma = params

        np_gt_data = np.array(gt_data)

        # Run transform
        np_gt_data = self.elastic_transform(np_gt_data,
                                            param_alpha, param_sigma)

        # TODO: CHeck with Andreanne
        if self.is3D:
            output_data = np_gt_data
        else:
            np_gt_data[np_gt_data >= 0.5] = 255.0
            np_gt_data[np_gt_data < 0.5] = 0.0
            np_gt_data = np_gt_data.astype(np.uint8)
            output_data = Image.fromarray(np_gt_data, mode='L')

        return output_data

    def __call__(self, sample):
        rdict = {}

        # Check probability of occurence
        if np.random.random() < self.p:
            input_data = sample['input']
            params = self.get_params(self.alpha_range,
                                     self.sigma_range)
            # TODO: decorator
            if isinstance(input_data, list):
                ret_input = [self.sample_augment(item, params)
                             for item in input_data]
            else:
                ret_input = self.sample_augment(input_data, params)

            # Update
            rdict['input'] = ret_input

            # Labeled data
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


class ToTensor3D(ToTensor):
    """This class extends ToTensor"""

    def undo_transform(self, sample):
        rdict = {}
        rdict['input'] = np.array(sample['input'])
        rdict['gt'] = np.array(sample['gt'])

        sample.update(rdict)
        return sample


class AdditiveGaussianNoise(IMEDTransform):

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


class Clahe(IMEDTransform):

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


class HistogramClipping(IMEDTransform):

    def __init__(self, min_percentile=5.0, max_percentile=95.0):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def do_clipping(self, data):
        data = np.copy(data)
        # Ensure that data is a numpy array
        data = np.array(data)
        # Run clipping
        percentile1 = np.percentile(data, self.min_percentile)
        percentile2 = np.percentile(data, self.max_percentile)
        data[data <= percentile1] = percentile1
        data[data >= percentile2] = percentile2
        return data

    def __call__(self, sample):
        input_data = sample['input']

        # TODO: Decorator?
        if isinstance(input_data, list):
            output_data = [self.do_clipping(data) for data in input_data]
        else:
            output_data = self.do_clipping(input_data)

        # Update
        rdict = {'input': output_data}
        sample.update(rdict)
        return sample

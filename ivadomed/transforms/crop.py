import numpy as np
from ivadomed.transforms.croppable_array import CroppableArray
from ivadomed.keywords import MetadataKW
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible


class Crop(ImedTransform):
    """Crop data.

    Args:
        size (tuple of int): Size of the output sample. Tuple of size 2 if dealing with 2D samples, 3 with 3D samples.

    Attributes:
        size (tuple of int): Size of the output sample. Tuple of size 3.
    """

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
    @multichannel_capable  # for multiple raters during training/preprocessing
    def __call__(self, sample, metadata):
        # Get params
        is_2d = sample.shape[-1] == 1
        th, tw, td = self.size
        fh, fw, fd, h, w, d = metadata[MetadataKW.CROP_PARAMS].get(self.__class__.__name__)

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
        fh, fw, fd, h, w, d = metadata[MetadataKW.CROP_PARAMS].get(self.__class__.__name__)

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


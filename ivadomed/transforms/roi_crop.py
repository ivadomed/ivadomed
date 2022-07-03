from scipy.ndimage import center_of_mass
from ivadomed.transforms.crop import Crop
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible
from ivadomed.keywords import MetadataKW


class ROICrop(Crop):
    """Make a crop of a specified size around a Region of Interest (ROI)."""

    @multichannel_capable
    @multichannel_capable  # for multiple raters during training/preprocessing
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # If crop_params are not in metadata,
        # then we are here dealing with ROI data to determine crop params
        if self.__class__.__name__ not in metadata[MetadataKW.CROP_PARAMS]:
            # Compute center of mass of the ROI
            h_roi, w_roi, d_roi = center_of_mass(sample.astype(int))
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
            metadata[MetadataKW.CROP_PARAMS][self.__class__.__name__] = params

        # Call base method
        return super().__call__(sample, metadata)
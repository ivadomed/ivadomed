from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible
from ivadomed.keywords import MetadataKW
from ivadomed.transforms.crop import Crop


class CenterCrop(Crop):
    """Make a centered crop of a specified size."""

    @multichannel_capable
    @multichannel_capable  # for multiple raters during training/preprocessing
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # Crop parameters
        th, tw, td = self.size
        h, w, d = sample.shape
        fh = int(round((h - th) / 2.))
        fw = int(round((w - tw) / 2.))
        fd = int(round((d - td) / 2.))
        params = (fh, fw, fd, h, w, d)
        metadata[MetadataKW.CROP_PARAMS][self.__class__.__name__] = params

        # Call base method
        return super().__call__(sample, metadata)

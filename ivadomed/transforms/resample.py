from scipy.ndimage import zoom
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.keywords import MetadataKW
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible


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
        """Resample to original resolution."""
        assert MetadataKW.DATA_SHAPE in metadata
        is_2d = sample.shape[-1] == 1

        # Get params
        original_shape = metadata[MetadataKW.PRE_RESAMPLE_SHAPE]
        current_shape = sample.shape
        params_undo = [x / y for x, y in zip(original_shape, current_shape)]
        if is_2d:
            params_undo[-1] = 1.0

        # Undo resampling
        data_out = zoom(sample,
                        zoom=params_undo,
                        order=1 if metadata[MetadataKW.DATA_TYPE] == 'gt' else 2)

        # Data type
        data_out = data_out.astype(sample.dtype)

        return data_out, metadata

    @multichannel_capable
    @multichannel_capable  # for multiple raters during training/preprocessing
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        """Resample to a given resolution, in millimeters."""
        # Get params
        # Voxel dimension in mm
        is_2d = sample.shape[-1] == 1
        metadata[MetadataKW.PRE_RESAMPLE_SHAPE] = sample.shape
        # metadata is not a dictionary!
        zooms = list(metadata[MetadataKW.ZOOMS])

        if len(zooms) == 2:
            zooms += [1.0]

        hfactor = zooms[0] / self.hspace
        wfactor = zooms[1] / self.wspace
        dfactor = zooms[2] / self.dspace
        params_resample = (hfactor, wfactor, dfactor) if not is_2d else (hfactor, wfactor, 1.0)

        # Run resampling
        data_out = zoom(sample,
                        zoom=params_resample,
                        order=1 if metadata[MetadataKW.DATA_TYPE] == 'gt' else 2)

        # Data type
        data_out = data_out.astype(sample.dtype)

        return data_out, metadata


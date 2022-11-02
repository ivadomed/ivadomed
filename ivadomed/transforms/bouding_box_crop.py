from ivadomed.transforms.crop import Crop
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible
from ivadomed.keywords import MetadataKW


class BoundingBoxCrop(Crop):
    """Crops image according to given bounding box."""

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata):
        assert MetadataKW.BOUNDING_BOX in metadata
        x_min, x_max, y_min, y_max, z_min, z_max = metadata[MetadataKW.BOUNDING_BOX]
        x, y, z = sample.shape
        metadata[MetadataKW.CROP_PARAMS][self.__class__.__name__] = (x_min, y_min, z_min, x, y, z)

        # Call base method
        return super().__call__(sample, metadata)
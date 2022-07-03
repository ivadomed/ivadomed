from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable


class NormalizeInstance(ImedTransform):
    """Normalize a tensor or an array image with mean and standard deviation estimated from the sample itself."""

    @multichannel_capable
    def undo_transform(self, sample, metadata=None):
        # Nothing
        return sample, metadata

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        data_out = (sample - sample.mean()) / sample.std()
        return data_out, metadata

import numpy as np
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible
from ivadomed.keywords import MetadataKW


class RandomReverse(ImedTransform):
    """Make a randomized symmetric inversion of the different values of each dimensions."""

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        if MetadataKW.REVERSE in metadata:
            flip_axes = metadata[MetadataKW.REVERSE]
        else:
            # Flip axis booleans
            flip_axes = [np.random.randint(2) == 1 for _ in [0, 1, 2]]
            # Save in metadata
            metadata[MetadataKW.REVERSE] = flip_axes

        # Run flip
        for idx_axis, flip_bool in enumerate(flip_axes):
            if flip_bool:
                sample = np.flip(sample, axis=idx_axis).copy()

        return sample, metadata

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        assert MetadataKW.REVERSE in metadata
        return self.__call__(sample, metadata)
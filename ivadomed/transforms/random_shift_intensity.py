import numpy as np
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable
from ivadomed.keywords import MetadataKW


class RandomShiftIntensity(ImedTransform):
    """Add a random intensity offset.

    Args:
        shift_range (tuple of floats): Tuple of length two. Specifies the range where the offset that is applied is
            randomly selected from.
        prob (float): Between 0 and 1. Probability of occurence of this transformation.
    """

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
        metadata[MetadataKW.OFFSET] = offset
        # Shift intensity
        data = (sample + offset).astype(sample.dtype)
        return data, metadata

    @multichannel_capable
    def undo_transform(self, sample, metadata=None):
        assert MetadataKW.OFFSET in metadata
        # Get offset
        offset = metadata[MetadataKW.OFFSET]
        # Substract offset
        data = (sample - offset).astype(sample.dtype)
        return data, metadata


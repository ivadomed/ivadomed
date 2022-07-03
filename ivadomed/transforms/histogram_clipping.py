import numpy as np
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable


class HistogramClipping(ImedTransform):
    """Performs intensity clipping based on percentiles.

    Args:
        min_percentile (float): Between 0 and 100. Lower clipping limit.
        max_percentile (float): Between 0 and 100. Higher clipping limit.
    """

    def __init__(self, min_percentile=5.0, max_percentile=95.0):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        data = np.copy(sample)
        # Run clipping
        percentile1 = np.percentile(sample, self.min_percentile)
        percentile2 = np.percentile(sample, self.max_percentile)
        data[sample <= percentile1] = percentile1
        data[sample >= percentile2] = percentile2
        return data, metadata


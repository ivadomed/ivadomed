import numpy as np
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable
from ivadomed.keywords import MetadataKW


class AdditiveGaussianNoise(ImedTransform):
    """Adds Gaussian Noise to images.

    Args:
        mean (float): Gaussian noise mean.
        std (float): Gaussian noise standard deviation.
    """

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        if MetadataKW.GAUSSIAN_NOISE in metadata:
            noise = metadata[MetadataKW.GAUSSIAN_NOISE]
        else:
            # Get random noise
            noise = np.random.normal(self.mean, self.std, sample.shape)
            noise = noise.astype(np.float32)

        # Apply noise
        data_out = sample + noise

        return data_out.astype(sample.dtype), metadata
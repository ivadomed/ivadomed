import numpy as np
from scipy.ndimage import gaussian_filter
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible
from ivadomed.keywords import MetadataKW


class RandomBlur(ImedTransform):
    """Applies a random blur to the image

        Args:
            sigma_range (tuple of floats): Standard deviation range for the gaussian filter
            p (float): Probability of performing blur
        """

    def __init__(self, sigma_range, p=0.5):
        self.sigma_range = sigma_range
        self.p = p

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        if np.random.random() < self.p:
            # Get params
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            # Save params
            metadata[MetadataKW.BLUR] = [sigma]

        else:
            metadata[MetadataKW.BLUR] = [None]

        if any(metadata[MetadataKW.BLUR]):
            # Apply random blur
            data_out = gaussian_filter(sample, sigma)

            # Keep data type
            data_out = data_out.astype(sample.dtype)

            return data_out, metadata

        else:
            return sample, metadata


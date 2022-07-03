import numpy as np
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible
from ivadomed.keywords import MetadataKW


class RandomGamma(ImedTransform):
    """Randomly changes the contrast of an image by gamma exponential

    Args:
        log_gamma_range (tuple of floats): Log gamma range for changing contrast. Length equals 2.
        p (float): Probability of performing the gamma contrast
    """

    def __init__(self, log_gamma_range, p=0.5):
        self.log_gamma_range = log_gamma_range
        self.p = p

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        if np.random.random() < self.p:
            # Get params
            gamma = np.exp(np.random.uniform(self.log_gamma_range[0], self.log_gamma_range[1]))

            # Save params
            metadata[MetadataKW.GAMMA] = [gamma]

        else:
            metadata[MetadataKW.GAMMA] = [None]

        if any(metadata[MetadataKW.GAMMA]):
            # Suppress the overflow case (due to exponentiation)
            with np.errstate(over='ignore'):
                # Apply gamma contrast
                data_out = np.sign(sample) * (np.abs(sample) ** gamma)

                # Keep data type
                data_out = data_out.astype(sample.dtype)

                # Clip +/- inf values to the max/min quantization of the native dtype
                data_out = np.nan_to_num(data_out)

            return data_out, metadata

        else:
            return sample, metadata


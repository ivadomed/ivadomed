import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible
from ivadomed.keywords import MetadataKW


class ElasticTransform(ImedTransform):
    """Applies elastic transformation.

    .. seealso::
        Simard, Patrice Y., David Steinkraus, and John C. Platt. "Best practices for convolutional neural networks
        applied to visual document analysis." Icdar. Vol. 3. No. 2003. 2003.

    Args:
        alpha_range (tuple of floats): Deformation coefficient. Length equals 2.
        sigma_range (tuple of floats): Standard deviation. Length equals 2.
    """

    def __init__(self, alpha_range, sigma_range, p=0.1):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # if params already defined, i.e. sample is GT
        if MetadataKW.ELASTIC in metadata:
            alpha, sigma = metadata[MetadataKW.ELASTIC]

        elif np.random.random() < self.p:
            # Get params
            alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            # Save params
            metadata[MetadataKW.ELASTIC] = [alpha, sigma]

        else:
            metadata[MetadataKW.ELASTIC] = [None, None]

        if any(metadata[MetadataKW.ELASTIC]):
            # Get shape
            shape = sample.shape

            # Compute random deformation
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            dz = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            if shape[2] == 1:
                dz = 0  # No deformation along the last dimension
            x, y, z = np.meshgrid(np.arange(shape[0]),
                                  np.arange(shape[1]),
                                  np.arange(shape[2]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), \
                      np.reshape(y + dy, (-1, 1)), \
                      np.reshape(z + dz, (-1, 1))

            # Apply deformation
            data_out = map_coordinates(sample, indices, order=1, mode='reflect')
            # Keep input shape
            data_out = data_out.reshape(shape)
            # Keep data type
            data_out = data_out.astype(sample.dtype)

            return data_out, metadata

        else:
            return sample, metadata

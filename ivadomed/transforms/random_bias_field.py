import numpy as np
import torchio as tio
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible, tio_transform
from ivadomed.keywords import MetadataKW


class RandomBiasField(ImedTransform):
    """Applies a random MRI bias field artifact to the image via torchio.RandomBiasField()

        Args:
            coefficients (float): Maximum magnitude of polynomial coefficients
            order: Order of the basis polynomial functions
            p (float): Probability of applying the bias field
        """

    def __init__(self, coefficients, order, p=0.5):
        self.coefficients = coefficients
        self.order = order
        self.p = p

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        if np.random.random() < self.p:
            # Get params
            random_bias_field = tio.Compose([tio.RandomBiasField(coefficients=self.coefficients,
                                                                 order=self.order,
                                                                 p=self.p)])

            # Save params
            metadata[MetadataKW.BIAS_FIELD] = [random_bias_field]

        else:
            metadata[MetadataKW.BIAS_FIELD] = [None]

        if any(metadata[MetadataKW.BIAS_FIELD]):
            # Apply random bias field
            data_out, history = tio_transform(x=sample, transform=random_bias_field)

            # Keep data type
            data_out = data_out.astype(sample.dtype)

            # Update metadata to history
            metadata[MetadataKW.BIAS_FIELD] = [history]

            return data_out, metadata

        else:
            return sample, metadata


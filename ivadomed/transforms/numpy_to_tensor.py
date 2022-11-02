import numpy as np
import torch
from ivadomed.transforms.imed_transform import ImedTransform


class NumpyToTensor(ImedTransform):
    """Converts nd array to tensor object."""

    def undo_transform(self, sample, metadata=None):
        """Converts Tensor to nd array."""
        return list(sample.numpy()), metadata

    def __call__(self, sample, metadata=None):
        """Converts nd array to Tensor."""
        sample = np.array(sample)
        # Use np.ascontiguousarray to avoid axes permutations issues
        arr_contig = np.ascontiguousarray(sample, dtype=sample.dtype)
        return torch.from_numpy(arr_contig), metadata
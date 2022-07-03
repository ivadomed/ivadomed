from skimage.exposure import equalize_adapthist
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable


class Clahe(ImedTransform):
    """ Applies Contrast Limited Adaptive Histogram Equalization for enhancing the local image contrast.

    .. seealso::
       Zuiderveld, Karel. "Contrast limited adaptive histogram equalization." Graphics gems (1994): 474-485.

    Default values are based on:

    .. seealso::
       Zheng, Qiao, et al. "3-D consistent and robust segmentation of cardiac images by deep learning with spatial
       propagation." IEEE transactions on medical imaging 37.9 (2018): 2137-2148.

    Args:
        clip_limit (float): Clipping limit, normalized between 0 and 1.
        kernel_size (tuple of int): Defines the shape of contextual regions used in the algorithm. Length equals image
        dimension (ie 2 or 3 for 2D or 3D, respectively).
    """

    def __init__(self, clip_limit=3.0, kernel_size=(8, 8)):
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        assert len(self.kernel_size) == len(sample.shape)
        # Run equalization
        data_out = equalize_adapthist(sample,
                                      kernel_size=self.kernel_size,
                                      clip_limit=self.clip_limit).astype(sample.dtype)

        return data_out, metadata

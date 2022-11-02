from ivadomed.transforms.numpy_to_tensor import NumpyToTensor


class UndoCompose(object):
    """Undo the Compose transformations.

    Call the undo transformations in the inverse order than the "do transformations".

    Attributes:
        compose (torchvision_transforms.Compose):

    Args:
        transforms (torchvision_transforms.Compose):
    """

    def __init__(self, compose):
        self.transforms = compose

    def __call__(self, sample, metadata, data_type='gt'):
        if self.transforms.transform[data_type] is None:
            # In case self.transforms.transform[data_type] is None
            return None, None
        else:
            numpy_to_tensor = NumpyToTensor()
            sample, metadata = numpy_to_tensor.undo_transform(sample, metadata)
            for tr in self.transforms.transform[data_type].transforms[::-1]:
                sample, metadata = tr.undo_transform(sample, metadata)
            return sample, metadata

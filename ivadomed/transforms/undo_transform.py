class UndoTransform(object):
    """Call undo transformation.

    Attributes:
        transform (ImedTransform):

    Args:
        transform (ImedTransform):
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform.undo_transform(sample)
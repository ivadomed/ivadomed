class ImedTransform(object):
    """Base class for transforamtions."""

    def __call__(self, sample, metadata=None):
        raise NotImplementedError("You need to implement the transform() method.")

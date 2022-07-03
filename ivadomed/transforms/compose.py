from loguru import logger
from ivadomed.transforms.numpy_to_tensor import NumpyToTensor
from torchvision import transforms as torchvision_transforms


class Compose(object):
    """Composes transforms together.

    Composes transforms together and split between images, GT and ROI.

    self.transform is a dict:
        - keys: "im", "gt" and "roi"
        - values torchvision_transform.Compose objects.

    Attributes:
        dict_transforms (dict): Dictionary where the keys are the transform names
            and the value their parameters.
        requires_undo (bool): If True, does not include transforms which do not have an undo_transform
            implemented yet.

    Args:
        transform (dict): Keys are "im", "gt", "roi" and values are torchvision_transforms.Compose of the
            transformations of interest.
    """

    def __init__(self, dict_transforms, requires_undo=False):
        list_tr_im, list_tr_gt, list_tr_roi = [], [], []
        for transform in dict_transforms.keys():
            parameters = dict_transforms[transform]

            # Get list of data type
            if "applied_to" in parameters:
                list_applied_to = parameters["applied_to"]
            else:
                list_applied_to = ["im", "gt", "roi"]

            # call transform
            if transform in globals():
                if transform == "NumpyToTensor":
                    continue
                params_cur = {k: parameters[k] for k in parameters if k != "applied_to" and k != "preprocessing"}
                transform_obj = globals()[transform](**params_cur)
            else:
                raise ValueError('ERROR: {} transform is not available. '
                                 'Please check its compatibility with your model json file.'.format(transform))

            # check if undo_transform method is implemented
            if requires_undo:
                if not hasattr(transform_obj, 'undo_transform'):
                    logger.info('{} transform not included since no undo_transform available for it.'.format(transform))
                    continue

            if "im" in list_applied_to:
                list_tr_im.append(transform_obj)
            if "roi" in list_applied_to:
                list_tr_roi.append(transform_obj)
            if "gt" in list_applied_to:
                list_tr_gt.append(transform_obj)

        self.transform = {
            "im": torchvision_transforms.Compose(list_tr_im),
            "gt": torchvision_transforms.Compose(list_tr_gt),
            "roi": torchvision_transforms.Compose(list_tr_roi)}

    def __call__(self, sample, metadata, data_type='im', preprocessing=False):
        if self.transform[data_type] is None or len(metadata) == 0:
            # In case self.transform[data_type] is None
            return None, None
        else:
            for tr in self.transform[data_type].transforms:
                sample, metadata = tr(sample, metadata)

            if not preprocessing:
                numpy_to_tensor = NumpyToTensor()
                sample, metadata = numpy_to_tensor(sample, metadata)
            return sample, metadata
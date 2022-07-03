import numpy as np
import numbers
import random
import math
from scipy.ndimage import affine_transform
from ivadomed.transforms.imed_transform import ImedTransform
from ivadomed.transforms.utils import multichannel_capable, two_dim_compatible
from ivadomed.keywords import MetadataKW


class RandomAffine(ImedTransform):
    """Apply Random Affine transformation.

    Args:
        degrees (float): Positive float or list (or tuple) of length two. Angles in degrees. If only a float is
            provided, then rotation angle is selected within the range [-degrees, degrees]. Otherwise, the list / tuple
            defines this range.
        translate (list of float): List of floats between 0 and 1, of length 2 or 3 depending on the sample shape (2D
            or 3D). These floats defines the maximum range of translation along each axis.
        scale (list of float): List of floats between 0 and 1, of length 2 or 3 depending on the sample shape (2D
            or 3D). These floats defines the maximum range of scaling along each axis.

    Attributes:
        degrees (tuple of floats):
        translate (list of float):
        scale (list of float):
    """

    def __init__(self, degrees=0, translate=None, scale=None):
        # Rotation
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        # Scale
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and (len(scale) == 2 or len(scale) == 3), \
                "scale should be a list or tuple and it must be of length 2 or 3."
            for s in scale:
                if not (0.0 <= s <= 1.0):
                    raise ValueError("scale values should be between 0 and 1")
            if len(scale) == 2:
                scale.append(0.0)
            self.scale = scale
        else:
            self.scale = [0., 0., 0.]

        # Translation
        if translate is not None:
            assert isinstance(translate, (tuple, list)) and (len(translate) == 2 or len(translate) == 3), \
                "translate should be a list or tuple and it must be of length 2 or 3."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
            if len(translate) == 2:
                translate.append(0.0)
        self.translate = translate

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # Rotation
        # If angle and metadata have been already defined for this sample, then use them
        if MetadataKW.ROTATION in metadata:
            angle, axes = metadata[MetadataKW.ROTATION]
        # Otherwise, get random ones
        else:
            # Get the random angle
            angle = math.radians(np.random.uniform(self.degrees[0], self.degrees[1]))
            # Get the two axes that define the plane of rotation
            axes = list(random.sample(range(3 if sample.shape[2] > 1 else 2), 2))
            axes.sort()
            # Save params
            metadata[MetadataKW.ROTATION] = [angle, axes]

        # Scale
        if MetadataKW.SCALE in metadata:
            scale_x, scale_y, scale_z = metadata[MetadataKW.SCALE]
        else:
            scale_x = random.uniform(1 - self.scale[0], 1 + self.scale[0])
            scale_y = random.uniform(1 - self.scale[1], 1 + self.scale[1])
            scale_z = random.uniform(1 - self.scale[2], 1 + self.scale[2])
            metadata[MetadataKW.SCALE] = [scale_x, scale_y, scale_z]

        # Get params
        if MetadataKW.TRANSLATION in metadata:
            translations = metadata[MetadataKW.TRANSLATION]
        else:
            self.data_shape = sample.shape

            if self.translate is not None:
                max_dx = self.translate[0] * self.data_shape[0]
                max_dy = self.translate[1] * self.data_shape[1]
                max_dz = self.translate[2] * self.data_shape[2]
                translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                                np.round(np.random.uniform(-max_dy, max_dy)),
                                np.round(np.random.uniform(-max_dz, max_dz)))
            else:
                translations = (0, 0, 0)

            metadata[MetadataKW.TRANSLATION] = translations

        # Do rotation
        shape = 0.5 * np.array(sample.shape)
        if axes == [0, 1]:
            rotate = np.array([[math.cos(angle), -math.sin(angle), 0],
                               [math.sin(angle), math.cos(angle), 0],
                               [0, 0, 1]])
        elif axes == [0, 2]:
            rotate = np.array([[math.cos(angle), 0, math.sin(angle)],
                               [0, 1, 0],
                               [-math.sin(angle), 0, math.cos(angle)]])
        elif axes == [1, 2]:
            rotate = np.array([[1, 0, 0],
                               [0, math.cos(angle), -math.sin(angle)],
                               [0, math.sin(angle), math.cos(angle)]])
        else:
            raise ValueError("Unknown axes value")

        scale = np.array([[1 / scale_x, 0, 0], [0, 1 / scale_y, 0], [0, 0, 1 / scale_z]])
        if MetadataKW.UNDO in metadata and metadata[MetadataKW.UNDO]:
            transforms = scale.dot(rotate)
        else:
            transforms = rotate.dot(scale)

        offset = shape - shape.dot(transforms) + translations

        data_out = affine_transform(sample, transforms.T, order=1, offset=offset,
                                    output_shape=sample.shape).astype(sample.dtype)

        return data_out, metadata

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        assert MetadataKW.ROTATION in metadata
        assert MetadataKW.SCALE in metadata
        assert MetadataKW.TRANSLATION in metadata
        # Opposite rotation, same axes
        angle, axes = - metadata[MetadataKW.ROTATION][0], metadata[MetadataKW.ROTATION][1]
        scale = 1 / np.array(metadata[MetadataKW.SCALE])
        translation = - np.array(metadata[MetadataKW.TRANSLATION])

        # Undo rotation
        dict_params = {MetadataKW.ROTATION: [angle, axes], MetadataKW.SCALE: scale,
                       MetadataKW.TRANSLATION: [0, 0, 0], MetadataKW.UNDO: True}

        data_out, _ = self.__call__(sample, dict_params)

        data_out = affine_transform(data_out, np.identity(3), order=1, offset=translation,
                                    output_shape=sample.shape).astype(sample.dtype)

        return data_out, metadata
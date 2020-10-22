#!/usr/bin/env python

import os
import argparse
import nibabel as nib
import numpy as np
import random
import json

from ivadomed import config_manager as imed_config_manager
from ivadomed.loader import utils as imed_loader_utils
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed import maths as imed_maths


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input image filename.")
    parser.add_argument("-c", "--config", required=True,
                        help="Config filename.")
    parser.add_argument("-n", "--number", required=False, default=1,
                        help="Number of random slices to visualize.")
    parser.add_argument("-o", "--output", required=False, default="./",
                        help="Output folder.")
    parser.add_argument("-r", "--roi", required=False,
                        help="ROI filename. Only required if ROICrop is part of the transformations.")
    return parser


def get_data(fname_in, axis):
    """Get data from fname along an axis.

    Args:
         fname_in string: image fname
         axis int:

    Returns:
        nibabel, ndarray
    """
    # Load image
    input_img = nib.load(fname_in)
    # Reorient as canonical
    input_img = nib.as_closest_canonical(input_img)
    # Get input data
    input_data = input_img.get_fdata(dtype=np.float32)
    # Reorient data
    input_data = imed_loader_utils.orient_img_hwd(input_data, slice_axis=axis)
    return input_img, input_data


def run_visualization(input, config, number, output, roi):
    """Utility function to visualize Data Augmentation transformations.

    Data augmentation is a key part of the Deep Learning training scheme. This script aims at facilitating the
    fine-tuning of data augmentation parameters. To do so, this script provides a step-by-step visualization of the
    transformations that are applied on data.

    This function applies a series of transformations (defined in a configuration file
    ``-c``) to ``-n`` 2D slices randomly extracted from an input image (``-i``), and save as png the resulting sample
    after each transform.

    For example::

        ivadomed_visualize_transforms -i t2s.nii.gz -n 1 -c config.json -r t2s_seg.nii.gz

    Provides a visualization of a series of three transformation on a randomly selected slice:

    .. image:: ../../images/transforms_im.png
        :width: 600px
        :align: center

    And on a binary mask::

        ivadomed_visualize_transforms -i t2s_gmseg.nii.gz -n 1 -c config.json -r t2s_seg.nii.gz

    Gives:

    .. image:: ../../images/transforms_gt.png
        :width: 600px
        :align: center

    Args:
         input (string): Image filename. Flag: ``--input``, ``-i``
         config (string): Configuration file filename. Flag: ``--config``, ``-c``
         number (int): Number of slices randomly extracted. Flag: ``--number``, ``-n``
         output (string): Folder path where the results are saved. Flag: ``--ofolder``, ``-o``
         roi (string): Filename of the region of interest. Only needed if ROICrop is part of the transformations.
                       Flag: ``--roi``, ``-r``
    """
    # Load context
    context = imed_config_manager.ConfigurationManager(config).get_config()

    # Create output folder
    if not os.path.isdir(output):
        os.makedirs(output)

    # Slice extracted according to below axis
    axis = imed_utils.AXIS_DCT[context["loader_parameters"]["slice_axis"]]
    # Get data
    input_img, input_data = get_data(input, axis)
    # Image or Mask
    is_mask = np.array_equal(input_data, input_data.astype(bool))
    # Get zooms
    zooms = imed_loader_utils.orient_shapes_hwd(input_img.header.get_zooms(), slice_axis=axis)
    # Get indexes
    indexes = random.sample(range(0, input_data.shape[2]), number)

    # Get training transforms
    training_transforms, _, _ = imed_transforms.get_subdatasets_transforms(context["transformation"])

    if "ROICrop" in training_transforms:
        if roi and os.path.isfile(roi):
            roi_img, roi_data = get_data(roi, axis)
        else:
            raise ValueError("\nPlease provide ROI image (-r) in order to apply ROICrop transformation.")

    # Compose transforms
    dict_transforms = {}
    stg_transforms = ""
    for transform_name in training_transforms:
        # We skip NumpyToTensor transform since that s only a change of data type
        if transform_name == "NumpyToTensor":
            continue

        # Update stg_transforms
        stg_transforms += transform_name + "_"

        # Add new transform to Compose
        dict_transforms.update({transform_name: training_transforms[transform_name]})
        composed_transforms = imed_transforms.Compose(dict_transforms)

        # Loop across slices
        for i in indexes:
            data = [input_data[:, :, i]]
            # Init metadata
            metadata = imed_loader_utils.SampleMetadata({"zooms": zooms, "data_type": "gt" if is_mask else "im"})

            # Apply transformations to ROI
            if "CenterCrop" in training_transforms or ("ROICrop" in training_transforms and os.path.isfile(roi)):
                metadata.__setitem__('crop_params', {})

            # Apply transformations to image
            stack_im, _ = composed_transforms(sample=data,
                                              metadata=[metadata for _ in range(number)],
                                              data_type="im")

            # Plot before / after transformation
            fname_out = os.path.join(output, stg_transforms+"slice"+str(i)+".png")
            print("Fname out: {}.".format(fname_out))
            print("\t{}".format(dict(metadata)))
            # rescale intensities
            if len(stg_transforms[:-1].split("_")) == 1:
                before = np.rot90(imed_maths.rescale_values_array(data[0], 0.0, 1.0))
            else:
                before = after
            after = np.rot90(imed_maths.rescale_values_array(stack_im[0], 0.0, 1.0))
            # Plot
            imed_utils.plot_transformed_sample(before,
                                               after,
                                               list_title=["\n".join(stg_transforms[:-1].split("_")[:-1]),
                                                           "\n".join(stg_transforms[:-1].split("_"))],
                                               fname_out=fname_out,
                                               cmap="jet" if is_mask else "gray")


def main():
    imed_utils.init_ivadomed()

    parser = get_parser()
    args = parser.parse_args()
    input = args.input
    config = args.config
    number = int(args.number)
    output = args.output
    roi = args.roi
    # Run script
    run_visualization(input, config, number, output, roi)


if __name__ == '__main__':
    main()

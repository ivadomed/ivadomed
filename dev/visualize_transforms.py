#!/usr/bin/env python
##############################################################
#
# This script apply a series of transforms to 2D slices extracted from an input image,
#   and save as png the resulting sample after each transform.
#
# Usage: python dev/visualize_transforms.py -i <input_filename> -a <int>
#
##############################################################

import argparse
import nibabel as nib
import numpy as np

from ivadomed.loader import utils as imed_loader_utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input image filename.")
    parser.add_argument("-a", "--axis", required=True, type=int,
                        help="Slice axis for slice extraction: 0 for sagittal, 1 for coronal, 2 for axial.")
    return parser


def run_visualization(args):
    """Run visualization. Main function of this script.

    Args:
         args argparse.ArgumentParser:

    Returns:
        None
    """
    # Get params
    fname_input = args.i
    axis = args.a
    # Load image
    input_img = nib.load(fname_input)
    # Reorient as canonical
    input_img = nib.as_closest_canonical(input_img)
    # Get input data
    input_data = input_img.get_fdata(dtype=np.float32)
    # Reorient data
    input_data = imed_loader_utils.orient_img_hwd(input_data, slice_axis=axis)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run_visualization(args)

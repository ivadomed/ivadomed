#!/usr/bin/env python
##############################################################
#
# This script apply a series of transforms to 2D slices extracted from an input image,
#   and save as png the resulting sample after each transform.
#
# Usage: python dev/visualize_transforms.py -i <input_filename> -a <int> -c <fname_config> -n <int>
#
##############################################################

import argparse
import nibabel as nib
import numpy as np
import random
import json

from ivadomed.loader import utils as imed_loader_utils
from ivadomed import transforms as imed_transforms

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input image filename.")
    parser.add_argument("-c", "--config", required=True,
                        help="Config filename.")
    parser.add_argument("-n", "--number", required=False, default=1,
                        help="Number of random slices to visualize.")
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
    n_slices = args.n
    with open(args.c, "r") as fhandle:
        context = json.load(fhandle)

    # Load image
    input_img = nib.load(fname_input)
    # Reorient as canonical
    input_img = nib.as_closest_canonical(input_img)
    # Get input data
    input_data = input_img.get_fdata(dtype=np.float32)
    # Reorient data
    input_data = imed_loader_utils.orient_img_hwd(input_data, slice_axis=axis)
    # Get indexes
    indexes = random.sample(range(0, input_data.shape[2]), n_slices)
    # Get slices list
    list_data = [np.expand_dims(input_data[:, :, i], axis=0) for i in indexes]

    # Compose transforms
    transforms = imed_transforms.Compose(context["transformation"])




if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run_visualization(args)

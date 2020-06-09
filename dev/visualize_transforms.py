#!/usr/bin/env python
##############################################################
#
# This script apply a series of transforms to 2D slices extracted from an input image,
#   and save as png the resulting sample after each transform.
#
# Usage: python dev/visualize_transforms.py -i <input_filename>
#
##############################################################

import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input image filename.")
    return parser


def run_visualization(args):
    """Run visualization. Main function of this script.

    Args:
         args argparse.ArgumentParser:

    Returns:
        None
    """
    fname_input = args.i


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run_visualization(args)

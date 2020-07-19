#!/usr/bin/env python

import os
import argparse
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input log directory.")
    return parser


def run_plot_training_curves(input_folder):
    """Utility function to XX.

    Data augmentation is a key part of the Deep Learning training scheme. This script aims at facilitating the
    fine-tuning of data augmentation parameters. To do so, this script provides a step-by-step visualization of the
    transformations that are applied on data.

    This function applies a series of transformations (defined in a configuration file
    ``-c``) to ``-n`` 2D slices randomly extracted from an input image (``-i``), and save as png the resulting sample
    after each transform.

    For example::

        ivadomed_XX

    .. image:: ../../images/XX
        :width: 600px
        :align: center

    Args:
         input_folder (string): Log directory name. Flag: --input, -i
    """
    pass


def main():
    parser = get_parser()
    args = parser.parse_args()
    input_folder = args.input
    # Run script
    run_plot_training_curves(input_folder)


if __name__ == '__main__':
    main()

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

    XX

    For example::

        ivadomed_XX

    XX

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

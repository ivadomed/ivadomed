#!/usr/bin/env python

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input log directory.")
    return parser



def is_interesting_tag(tag):
    if 'val' in tag or 'train' in tag:
        return True
    else:
        return False

def parse_events_file(path: str) -> pd.DataFrame:
    metrics = defaultdict(list)
    for e in tf.train.summary_iterator(path):
        for v in e.summary.value:

            if isinstance(v.simple_value, float) and is_interesting_tag(v.tag):
                metrics[v.tag].append(v.simple_value)
            if v.tag == 'loss' or v.tag == 'accuracy':
                print(v.simple_value)
    metrics_df = pd.DataFrame({k: v for k, v in metrics.items() if len(v) > 1})
    return metrics_df


def find_events(input_folder):
    """Get TF events path from input_folder.

    Args:
        input_folder (str): Input folder path.
    Returns:
        dict: keys are subfolder names and values are events' paths.
    """
    dict = {}
    for fold in os.listdir(input_folder):
        fold_path = os.path.join(input_folder, fold)
        if os.path.isdir(fold_path):
            event_list = [f for f in os.listdir(fold_path) if f.startswith("events.out.tfevents.")]
            if len(event_list):
                if len(event_list) > 1:
                    print('Multiple events found in this folder: {}.\nPlease keep only one before running '
                          'this script again.'.format(fold_path))
                dict[fold] = os.path.join(input_folder, fold, event_list[0])
    return dict


def plot_curve(event_folder, event_path, fname_out):
    """Plot event curve.

    Args:
        event_folder (str): Event folder name.
        event_path (str): Event path.
        fname_out (str): Filename for the plot.
    """
    for e in summary_iterator(event_path):
        for v in e.summary.value:
            print(v.tag)

            #if isinstance(v.simple_value, float) and is_interesting_tag(v.tag):
            #    metrics[v.tag].append(v.simple_value)
            #if v.tag == 'loss' or v.tag == 'accuracy':
            #    print(v.simple_value)

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
    # Find tf folders
    events_dict = find_events(input_folder)

    # Iterate through the events
    for event in events_dict:
        fname_out = os.path.join(input_folder, event, "plot.png")
        plot_curve(event, events_dict[event], fname_out)




def main():
    parser = get_parser()
    args = parser.parse_args()
    input_folder = args.input
    # Run script
    run_plot_training_curves(input_folder)


if __name__ == '__main__':
    main()

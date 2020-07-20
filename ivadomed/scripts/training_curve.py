#!/usr/bin/env python

import os
import argparse
from collections import defaultdict
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import matplotlib.pyplot as plt

DEBUGGING = False

if DEBUGGING:
    import matplotlib
    matplotlib.use('TkAgg')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Input log directory.")
    parser.add_argument("-m", "--multiple", required=False, default=False, type=bool,
                        help="If True, then multiple log directories are considered: all available folders with -i as "
                             "prefix.")
    parser.add_argument("-o", "--output", required=False, type=str,
                        help="Output folder. If not specified, results are saved under "
                             "input_folder/plot_training_curves.")
    return parser


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


def get_data(event_dict):
    """Get data as Pandas dataframe.

    Args:
        event_dict (dict): Dictionary containing the TF event names and their paths.
    Returns:
        Pandas Dataframe: where the columns are the metrics or losses and the rows represent the epochs.
    """
    metrics = defaultdict(list)
    for tf_tag in event_dict:
        for e in summary_iterator(event_dict[tf_tag]):
            for v in e.summary.value:
                if isinstance(v.simple_value, float):
                    if tf_tag.startswith("Validation_Metrics_"):
                        tag = tf_tag.split("Validation_Metrics_")[1]
                    elif tf_tag.startswith("losses_"):
                        tag = tf_tag.split("losses_")[1]
                    else:
                        print("Unknown TF tag: {}.".format(tf_tag))
                        exit()
                    metrics[tag].append(v.simple_value)
    metrics_df = pd.DataFrame.from_dict(metrics)
    return metrics_df


def plot_curve(data, y_label, fname_out):
    """Plot curve of metrics or losses for each epoch.

    Args:
        data (pd.DataFrame):
        y_label (str): Label for the y-axis.
        fname_out (str): Save plot with this filename.
    """
    # Create count of the number of epochs
    epoch_count = range(1, len(data) + 1)

    plt.figure(figsize=(10, 5))

    for k in data.keys():
        plt.plot(epoch_count, data[k].tolist())

    plt.legend(data.keys(), loc="best")
    plt.grid(linestyle='dotted')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.xlim([1, len(data)])

    if DEBUGGING:
        plt.show()
        exit()
    else:
        plt.savefig(fname_out)


def run_plot_training_curves(input_folder, output_folder, multiple_training=False):
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
    # Find training folders:
    if multiple_training:
        prefix = input_folder.split('/')[-1]
        input_folder = '/'.join(input_folder.split('/')[:-1])
        input_folder_list = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.startswith(prefix)]
    else:
        input_folder_list = [input_folder]

    events_df_list = []
    for log_directory in input_folder_list:
        # Find tf folders
        events_dict = find_events(log_directory)

        # Get data as dataframe
        events_vals_df = get_data(events_dict)

        # Store data
        events_df_list.append(events_vals_df)

    # Create output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, "plot_training_curves")
    if os.path.isdir(output_folder):
        print("Output folder already exists: {}.".format(output_folder))
    else:
        print("Creating output folder: {}.".format(output_folder))
        os.makedirs(output_folder)

    # Plot train and valid losses together
    fname_out = os.path.join(output_folder, "losses.png")
    loss_keys = [k for k in events_vals_df.keys() if k.endswith("loss")]
    plot_curve(events_vals_df[loss_keys], "loss", fname_out)

    # Plot each validation metric separetly
    for tag in events_vals_df.keys():
        if not tag.endswith("loss"):
            fname_out = os.path.join(output_folder, tag+".png")
            plot_curve(events_vals_df[[tag]], tag, fname_out)


def main():
    parser = get_parser()
    args = parser.parse_args()
    input_folder = args.input
    multiple = args.multiple
    output_folder = args.output
    # Run script
    run_plot_training_curves(input_folder, output_folder, multiple)


if __name__ == '__main__':
    main()

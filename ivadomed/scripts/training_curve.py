#!/usr/bin/env python

import os
import argparse
import numpy as np
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
                        help="Input log directory. If using -m True, this parameter indicates the suffix path of all "
                             "log directories of interest.")
    parser.add_argument("--multiple", required=False, dest="multiple", action='store_true',
                        help="Multiple log directories are considered: all available folders with -i as "
                             "prefix. The plot represents the mean value (hard line) surrounded by the standard "
                             "deviation envelope.")
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


def plot_curve(data_list, y_label, fname_out):
    """Plot curve of metrics or losses for each epoch.

    Args:
        data_list (list): list of pd.DataFrame, one for each log_directory
        y_label (str): Label for the y-axis.
        fname_out (str): Save plot with this filename.
    """
    # Create count of the number of epochs
    max_nb_epoch = max([len(data_list[i]) for i in range(len(data_list))])
    epoch_count = range(1, max_nb_epoch + 1)

    for k in data_list[0].keys():
        data_k = pd.concat([data_list[i][k] for i in range(len(data_list))], axis=1)
        mean_data_k = data_k.mean(axis=1, skipna=True)
        std_data_k = data_k.std(axis=1, skipna=True)
        std_minus_data_k = (mean_data_k - std_data_k).tolist()
        std_plus_data_k = (mean_data_k + std_data_k).tolist()
        mean_data_k = mean_data_k.tolist()
        plt.plot(epoch_count, mean_data_k)
        plt.fill_between(epoch_count, std_minus_data_k, std_plus_data_k, alpha=0.3)

    plt.legend(data_list[0].keys(), loc="best")
    plt.grid(linestyle='dotted')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.xlim([1, max_nb_epoch])

    if DEBUGGING:
        plt.show()
        exit()
    else:
        plt.savefig(fname_out)


def run_plot_training_curves(input_folder, output_folder, multiple_training=False):
    """Utility function to plot the training curves.

    This function uses the TensorFlow summary that is generated during a training to plot for each epoch:
        - the training against the validation loss
        - the metrics computed on the validation sub-dataset.

    It could consider one log directory at a time, for example::
    .. image:: ../../images/plot_loss_single.png
        :width: 600px
        :align: center

    ... or multiple (using ``multiple_training=True``). In that case, the hard line represent the mean value across the
    trainings whereas the envelope represents the standard deviation::
    .. image:: ../../images/plot_loss_multiple.png
        :width: 600px
        :align: center

    Args:
        input_folder (str): Log directory name. Flag: --input, -i. If using ``-m True``, this parameter indicates the
            suffix path of all log directories of interest.
        output_folder (str): Output folder. Flag: --output, -o. If not specified, results are saved under
            ``input_folder/plot_training_curves``.
        multiple_training (bool): Indicates if multiple log directories are considered (``True``) or not (``False``).
            Flag: --multiple, -m. If ``True``, all available folders with ``-i`` as prefix. The plot represents the mean
            value (hard line) surrounded by the standard deviation (envelope).
    """
    group_list = input_folder.split(",")

    # Init plot
    n_cols = 2
    n_rows = int(np.ceil(n_cols / float(n_cols)))
    plt.figure(figsize=(10 * n_cols, 5 * n_rows))

    for input_folder in group_list:
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
        loss_keys = [k for k in events_df_list[0].keys() if k.endswith("loss")]
        plot_curve([df[loss_keys] for df in events_df_list], "loss", fname_out)

        # Plot each validation metric separetly
        for tag in events_df_list[0].keys():
            if not tag.endswith("loss"):
                fname_out = os.path.join(output_folder, tag+".png")
                plot_curve([df[[tag]] for df in events_df_list], tag, fname_out)


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

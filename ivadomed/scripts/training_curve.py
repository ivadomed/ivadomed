#!/usr/bin/env python

import os
import argparse
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ivadomed import utils as imed_utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="""Input path. If using --multiple, this parameter indicates
                                the suffix path of all log directories of interest. To compare
                                trainings or set of trainings (using ``--multiple``) with subplots,
                                please list the paths by separating them with commas, e.g.
                                path_output1,path_output2.""",
                        metavar=imed_utils.Metavar.str)
    parser.add_argument("--multiple", required=False, dest="multiple", action='store_true',
                        help="""Multiple log directories are considered: all available folders
                                with -i as prefix. The plot represents the mean value (hard line)
                                surrounded by the standard deviation envelope.""")
    parser.add_argument("-y", "--ylim_loss", required=False, type=str,
                        help="""Indicates the limits on the y-axis for the loss plots, otherwise
                                these limits are automatically defined. Please separate the lower
                                and the upper limit by a comma, e.g. -1,0. Note: for the validation
                                metrics: the y-limits are always 0.0 and 1.0.""",
                        metavar=imed_utils.Metavar.float)
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="Output folder.", metavar=imed_utils.Metavar.file)
    return parser


def check_events_numbers(input_folder):
    """Check to make sure there is at most one summary in any folder or any subfolder.

    A summary is defined as any file of the format ``events.out.tfevents.{...}```

    Args:
        input_folder (str): Input folder path.
    """
    for fold in os.listdir(input_folder):
        fold_path = os.path.join(input_folder, fold)
        if os.path.isdir(fold_path):
            event_list = [f for f in os.listdir(fold_path) if f.startswith("events.out.tfevents.")]
            if len(event_list):
                if len(event_list) > 1:
                    raise ValueError(f"Multiple summary found in this folder: {fold_path}.\n"
                                     f"Please keep only one before running this script again.")


def plot_curve(data_list, y_label, fig_ax, subplot_title, y_lim=None):
    """Plot curve of metrics or losses for each epoch.

    Args:
        data_list (list): list of pd.DataFrame, one for each path_output
        y_label (str): Label for the y-axis.
        fig_ax (plt.subplot):
        subplot_title (str): Title of the subplot
        y_lim (list): List of the lower and upper limits of the y-axis.
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
        fig_ax.plot(epoch_count, mean_data_k, )
        fig_ax.fill_between(epoch_count, std_minus_data_k, std_plus_data_k, alpha=0.3)

    fig_ax.legend(data_list[0].keys(), loc="best")
    fig_ax.grid(linestyle='dotted')
    fig_ax.set_xlabel('Epoch')
    fig_ax.set_ylabel(y_label)
    if y_lim is not None:
        fig_ax.set_ylim(y_lim)
    fig_ax.set_xlim([1, max_nb_epoch])
    fig_ax.title.set_text('\n'.join(wrap(subplot_title, 80)))


def run_plot_training_curves(input_folder, output_folder, multiple_training=False, y_lim_loss=None):
    """Utility function to plot the training curves.

    This function uses the TensorFlow summary that is generated during a training to plot for each epoch:

        - the training against the validation loss
        - the metrics computed on the validation sub-dataset.

    It could consider one output path at a time, for example:

    .. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/scripts/plot_loss_single.png
        :width: 600px
        :align: center

    ... or multiple (using ``multiple_training=True``). In that case, the hard line represents
    the mean value across the trainings whereas the envelope represents the standard deviation:

    .. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/scripts/plot_loss_multiple.png
        :width: 600px
        :align: center

    It is also possible to compare multiple trainings (or set of trainings) by listing them
    in ``-i``, separated by commas:

    .. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/scripts/plot_loss_mosaic.png
        :width: 600px
        :align: center

    Args:
        input_folder (str): Input path name. Flag: ``--input``, ``-i``. If using ``--multiple``,
            this parameter indicates the suffix path of all log directories of interest. To compare
            trainings or set of trainings (using ``--multiple``) with subplots, please list the
            paths by separating them with commas, e.g. path_output1, path_output2
        output_folder (str): Output folder. Flag: ``--output``, ``-o``.
        multiple_training (bool): Indicates if multiple log directories are considered (``True``)
            or not (``False``). Flag: ``--multiple``. All available folders with ``-i`` as prefix
            are considered. The plot represents the mean value (hard line) surrounded by the
            standard deviation (envelope).
        y_lim_loss (list): List of the lower and upper limits of the y-axis of the loss plot.
    """
    group_list = input_folder.split(",")
    plt_dict = {}

    # Create output folder
    if os.path.isdir(output_folder):
        print(f"Output folder already exists: {output_folder}.")
    else:
        print(f"Creating output folder: {output_folder}.")
        os.makedirs(output_folder)

    # Config subplots
    if len(group_list) > 1:
        n_cols = 2
        n_rows = int(np.ceil(len(group_list) / float(n_cols)))
    else:
        n_cols, n_rows = 1, 1

    for i_subplot, input_folder in enumerate(group_list):
        input_folder = os.path.expanduser(input_folder)
        # Find training folders:
        if multiple_training:
            prefix = str(input_folder.split('/')[-1])
            input_folder = '/'.join(input_folder.split('/')[:-1])
            input_folder_list = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                                 f.startswith(prefix)]
        else:
            prefix = str(input_folder.split('/')[-1])
            input_folder_list = [input_folder]

        events_df_list = []
        for path_output in input_folder_list:
            # Find tf folders
            check_events_numbers(path_output)

            # Get data as dataframe
            events_vals_df = tensorboard_retrieve_event(path_output)

            # Store data
            events_df_list.append(events_vals_df)

        # Plot train and valid losses together
        loss_keys = [k for k in events_df_list[0].keys() if k.endswith("loss")]
        if i_subplot == 0:  # Init plot
            plt_dict[os.path.join(output_folder, "losses.png")] = plt.figure(figsize=(10 * n_cols, 5 * n_rows))
        ax = plt_dict[os.path.join(output_folder, "losses.png")].add_subplot(n_rows, n_cols, i_subplot + 1)
        plot_curve([df[loss_keys] for df in events_df_list],
                   y_label="loss",
                   fig_ax=ax,
                   subplot_title=prefix,
                   y_lim=y_lim_loss)

        # Plot each validation metric separetly
        for tag in events_df_list[0].keys():
            if not tag.endswith("loss"):
                if i_subplot == 0:  # Init plot
                    plt_dict[os.path.join(output_folder, tag + ".png")] = plt.figure(figsize=(10 * n_cols, 5 * n_rows))
                ax = plt_dict[os.path.join(output_folder, tag + ".png")].add_subplot(n_rows, n_cols, i_subplot + 1)
                plot_curve(data_list=[df[[tag]] for df in events_df_list],
                           y_label=tag,
                           fig_ax=ax,
                           subplot_title=prefix,
                           y_lim=[0, 1])

    for fname_out in plt_dict:
        plt_dict[fname_out].savefig(fname_out)


def tensorboard_retrieve_event(path_output):
    """Retrieve data from tensorboard summary event.

    Args:
        path_output (str): output path where the event files are located

    Returns:
        df: a panda dataframe where the columns are the metric or loss and the row are the epochs.

    """
    # TODO : Find a way to not hardcode this list of metrics/loss
    # These list of metrics and losses are in the same order as in the training file (where they are written)
    list_metrics = ['dice_score', 'multiclass dice_score', 'hausdorff_score', 'precision_score',
                    'recall_score', 'specificity_score', 'intersection_over_union', 'accuracy_score']

    list_loss = ['train_loss', 'validation_loss']

    # Each element in the summary iterator represent an element (e.g., scalars, images..)
    # stored in the summary for all epochs in the form of event.
    summary_iterators = [EventAccumulator(os.path.join(path_output, dname)).Reload() for dname in os.listdir(path_output)]

    metrics = defaultdict(list)
    num_metrics = 0
    num_loss = 0

    for i in range(len(summary_iterators)):
        if summary_iterators[i].Tags()['scalars'] == ['Validation/Metrics']:
            # we create a empty list
            out = [0 for i in range(len(summary_iterators[i].Scalars("Validation/Metrics")))]
            # we ensure that value are append in the right order by looking at the step value
            # (which represents the epoch)
            for events in summary_iterators[i].Scalars("Validation/Metrics"):
                out[events.step - 1] = events.value
            # keys are the defined metrics
            metrics[list_metrics[num_metrics]] = out
            num_metrics += 1
        elif summary_iterators[i].Tags()['scalars'] == ['losses']:
            out = [0 for i in range(len(summary_iterators[i].Scalars("losses")))]
            # we ensure that value are append in the right order by looking at the step value
            # (which represents the epoch)
            for events in summary_iterators[i].Scalars("losses"):
                out[events.step - 1] = events.value
            metrics[list_loss[num_loss]] = out
            num_loss += 1

    if num_loss == 0 and num_metrics == 0:
        raise Exception('No metrics or losses found in the event')
    metrics_df = pd.DataFrame.from_dict(metrics)
    return metrics_df


def main(args=None):
    imed_utils.init_ivadomed()
    parser = get_parser()
    args = imed_utils.get_arguments(parser, args)
    y_lim_loss = [int(y) for y in args.ylim_loss.split(',')] if args.ylim_loss else None

    run_plot_training_curves(input_folder=args.input, output_folder=args.output,
                             multiple_training=args.multiple, y_lim_loss=y_lim_loss)


if __name__ == '__main__':
    main()

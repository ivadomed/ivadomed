###########################################################################################################
#
# This script makes a figure with violinplots and significance values to compare between models
#
#        python3 visualize_and_compare_testing_models.py --logfolders path/to/logfolder1 path/to/logfolder2
#                              --metric metric/to/use --metadata metadata/label string/to/match
###########################################################################################################


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import itertools
import seaborn as sns
from scipy.stats import ks_2samp
from ivadomed.utils import init_ivadomed
import argparse
matplotlib.use('TkAgg')  # This is needed for plotting through a CLI call

# Konstantinos Nasiotis 2020
#
# Dependency: sudo apt-get install python3-tk
# - needed for matplotlib visualization through a CLI call
# ----------------------------------------------------------------------------------------------------------------------#


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfolders", required=True, nargs="*", dest="logfolders",
                        help="List of log folders from different models.")
    parser.add_argument("--metric", default='dice_class0', nargs=1, type=str, dest="metric",
                        help="Metric from evaluation_3Dmetrics.csv to base the plots on.")
    parser.add_argument("--metadata", required=False,  nargs=2, type=str, dest="metadata",
                        help="Selection based on metadata from participants.tsv: (1) Label from column (2) string to match")
    return parser


def visualize_and_compare_models(logfolders, metric, metadata):
    """This function allows violinplots visualization of multiple evaluation models simultaneously and performs a
       Kolmogorov–Smirnov significance test between each combination of models.

    If only one model is selected as input, only the Violinplot will be presented (no test will be superimposed)

    Usage example::

        visualize_and_compare_testing_models.py --logfolders ~/logs/logs_T1w ~/logs/logs_T2w
                                                --metric dice_class0 --metadata pathology ms

    .. image:: ../../images/visualize_and_compare_models.png
            :width: 600px
            :align: center


    Args:
        logfolders (list): list of folders that contain the logs of the models to be compared, Flag: ``--logfolders``
        metric (str):      column of "results_eval/evaluation_3Dmetrics.csv" to be used on the plots (default: dice_class0),
                           Flag: ``--metric``
        metadata (list) - Optional:   Allows visualization of violinplots only from subjects that match the metadata criteria.
                           2 elements - (1) column label of the participants.tsv metadata so only subjects that belong to
                           that category will be used and (2) string to be matched, Flag: ``--metadata``
            Example::

                --metadata pathology ms
    """


    # access CLI options
    print("logfolders: %r" % logfolders)
    print("metric: %r" % metric)
    if metadata:
        print("metadata: %r" % metadata)

    # Do a quick check that all the required files are present
    for folder in logfolders:
        if not os.path.exists(os.path.join(folder, 'results_eval', 'evaluation_3Dmetrics.csv')):
            print('evaluation_3Dmetrics.csv file is not present within ' + os.path.join(folder, 'results_eval'))
            raise Exception('evaluation_3Dmetrics.csv missing')
        if not os.path.exists(os.path.join(folder, 'participants.tsv')):
            print('participants.tsv file is not present within ' + folder)
            raise Exception('participants.tsv missing')

    if len(logfolders) < 1:
        raise Exception('No folders were selected - Nothing to show')

    columnNames = ["EvaluationModel", metric]
    df = pd.DataFrame([], columns=columnNames)

    for folder in logfolders:
        result = pd.read_csv(os.path.join(folder, 'results_eval', 'evaluation_3Dmetrics.csv'))

        if metadata:
            participant_metadata = pd.read_table(os.path.join(folder, 'participants.tsv'), encoding="ISO-8859-1")
            # Select only the subjects that satisfy the --metadata input
            selected_subjects = participant_metadata[participant_metadata[metadata[0]] == metadata[1]]["participant_id"].tolist()

            # Now select only the scores from these subjects
            result_subject_ids = result["image_id"]
            result_subject_ids = [i.split('_', 1)[0] for i in result_subject_ids]  # Get rid of _T1w, _T2w etc.

            result = result.iloc[[i for i in range(len(result_subject_ids)) if result_subject_ids[i] in selected_subjects]]

            if result.empty:
                print('No subject meet the selected criteria - skipping plot for: ' + folder)

        if not result.empty:
            scores = result[metric]
            folders = [os.path.basename(os.path.normpath(folder))] * len(scores)
            combined = np.column_stack((folders, scores.astype(np.object, folders))).T
            singleFolderDF = pd.DataFrame(combined, columnNames).T
            df = df.append(singleFolderDF, ignore_index=True)

    nFolders = len(logfolders)
    combinedNumbers = list(itertools.combinations(range(nFolders), 2))
    combinedFolders = list(itertools.combinations(logfolders, 2))

    # Pandas annoying issues
    df[metric] = df[metric].astype('float64')

    if not df.empty:

        # Plot all violinplots
        sns.violinplot(x="EvaluationModel", y=metric, data=df, color="0.8", inner='quartile')
        sns.stripplot(x="EvaluationModel", y=metric, data=df, jitter=True, zorder=1)

        # Display the mean performance on top of every violinplot
        for i in range(len(logfolders)):
            # This will be used to plot the mean value on top of each individual violinplot
            temp = df[metric][df['EvaluationModel'] == os.path.basename(os.path.normpath(logfolders[i]))]
            plt.text(i, df[metric].max() + 0.07, str((100 * temp.mean()).round() / 100), ha='center', va='top',
                     color='r')

        if len(logfolders) > 1:
            # Perform a Kolmogorov–Smirnov test for all combinations of results & connect the corresponding Violinplots
            for i in range(len(combinedNumbers)):
                dataX = df[metric][df['EvaluationModel'] ==
                                        os.path.basename(os.path.normpath(combinedFolders[i][0]))]
                dataY = df[metric][df['EvaluationModel'] ==
                                        os.path.basename(os.path.normpath(combinedFolders[i][1]))]

                KStest = ks_2samp(dataX, dataY)

                x1, x2 = combinedNumbers[i]

                y, h, col = df[metric].min() - 0.06 - 0.03 * i, -0.01, 'k'
                plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)

                # Show if the differentiation of the distributions is :
                # Not significant: ns, significant: *, very significant: ***
                if KStest.pvalue >= 0.5:
                    plt.text((x1 + x2) * .5, y + h, "ns", ha='center', va='bottom', color=col)
                elif 0.5 > KStest.pvalue >= 0.01:
                    plt.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color='r')
                elif KStest.pvalue < 0.01:
                    plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color='r')

        if metadata:
            plt.title("Metric:  " + metric + "\nMetadata:  " + metadata[0] + ":" + metadata[1])
        else:
            plt.title("Metric:  " + metric)

        plt.grid()
        plt.show(block=True)

        print('success')

    else:
        print('No subjects meet the criteria selected for any model. '
              'Probably you need to change the --metadata / --metric selection')


def main():
    init_ivadomed()

    parser = get_parser()
    args = parser.parse_args()

    # Run automate training
    visualize_and_compare_models(args.logfolders, args.metric, args.metadata)


if __name__ == '__main__':
    main()

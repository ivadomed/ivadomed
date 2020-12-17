import matplotlib
matplotlib.use('TkAgg')  # This is needed for plotting through a CLI call
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import itertools
import seaborn as sns
from scipy.stats import ks_2samp
import argparse
import sys


# This function allows violinplots visualization of multiple evaluation models simultaneously and performs a
# Kolmogorov–Smirnov significance test between each combination of models.
# If only one model is selected as input, only the Violinplot will be presented (no test will be superimposed)

# Inputs:
# --listfolders: list of folders that contain the logs - space separated
# --metric: column of "results_eval/evaluation_3Dmetrics.csv" to be used on the plots e.g. dice_class0
# --metadata (optional): 2 elements - (1) column of the participants.tsv metadata so only subjects that belong to that
#                        category will be used and (2) string to be matched e.g. pathology ms

# Example calls from terminal:
# python3 visualize_and_compare_testing_models.py --listfolders ~/logs/logs_NO_FILM_sctUsers ~/logs/logs_onlyT1w
# or
# python3 visualize_and_compare_testing_models.py --listfolders /home/nas/Desktop/logs/logs_*
#           --metric dice_class0 --metadata pathology ms


# Konstantinos Nasiotis 2020


# Dependency: sudo apt-get install python3-tk
# - needed for matplotlib visualization through a CLI call
# ----------------------------------------------------------------------------------------------------------------------#

def main(argv):
    # defined command line options
    # this also generates --help and error handling
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--listfolders",
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=[],  # default if nothing is provided
    )
    CLI.add_argument(
        "--metric",
        nargs=1,
        type=str,
        default=["dice_class0"],
    )
    CLI.add_argument(
        "--metadata",
        nargs=2,
        type=str,
        default=[],
    )

    # parse the command line
    args = CLI.parse_args()
    args.metric = args.metric[0]

    # access CLI options
    print("listfolders: %r" % args.listfolders)
    print("metric: %r" % args.metric)
    if args.metadata != []:
        print("metadata: %r" % args.metadata)

    # Get the list
    logFoldersToCompare = args.listfolders
    # Do a quick check that all the required files are present
    for folder in logFoldersToCompare:
        if not os.path.exists(os.path.join(folder, 'results_eval', 'evaluation_3Dmetrics.csv')):
            print('evaluation_3Dmetrics.csv file is not present within ' + os.path.join(folder, 'results_eval'))
            raise Exception('evaluation_3Dmetrics.csv missing')
        if not os.path.exists(os.path.join(folder, 'participants.tsv')):
            print('participants.tsv file is not present within ' + folder)
            raise Exception('participants.tsv missing')

    if len(logFoldersToCompare) < 1:
        raise Exception('No folders were selected - Nothing to show')

    columnNames = ["EvaluationModel", args.metric]
    df = pd.DataFrame([], columns=columnNames)

    for folder in logFoldersToCompare:
        result = pd.read_csv(os.path.join(folder, 'results_eval', 'evaluation_3Dmetrics.csv'))

        if args.metadata:
            participant_metadata = pd.read_table(os.path.join(folder, 'participants.tsv'), encoding="ISO-8859-1")
            # Select only the subjects that satisfy the --metadata input
            selected_subjects = participant_metadata[participant_metadata[args.metadata[0]] == args.metadata[1]]["participant_id"].tolist()

            # Now select only the scores from these subjects
            result_subject_ids = result["image_id"]
            result_subject_ids = [i.split('_', 1)[0] for i in result_subject_ids]  # Get rid of _T1w, _T2w etc.

            result = result.iloc[[i for i in range(len(result_subject_ids)) if result_subject_ids[i] in selected_subjects]]

            if result.empty:
                print('No subject meet the selected criteria - skipping plot for: ' + folder)

        if not result.empty:
            scores = result[args.metric]
            folders = [os.path.basename(os.path.normpath(folder))] * len(scores)
            combined = np.column_stack((folders, scores.astype(np.object, folders))).T
            singleFolderDF = pd.DataFrame(combined, columnNames).T
            df = df.append(singleFolderDF, ignore_index=True)

    nFolders = len(logFoldersToCompare)
    combinedNumbers = list(itertools.combinations(range(nFolders), 2))
    combinedFolders = list(itertools.combinations(logFoldersToCompare, 2))

    # Pandas annoying issues
    df[args.metric] = df[args.metric].astype('float64')

    if not df.empty:

        # Plot all violinplots
        sns.violinplot(x="EvaluationModel", y=args.metric, data=df, color="0.8", inner='quartile')
        sns.stripplot(x="EvaluationModel", y=args.metric, data=df, jitter=True, zorder=1)

        # Display the mean performance on top of every violinplot
        for i in range(len(logFoldersToCompare)):
            # This will be used to plot the mean value on top of each individual violinplot
            temp = df[args.metric][df['EvaluationModel'] == os.path.basename(os.path.normpath(logFoldersToCompare[i]))]
            plt.text(i, df[args.metric].max() + 0.07, str((100 * temp.mean()).round() / 100), ha='center', va='top',
                     color='r')

        if len(logFoldersToCompare) > 1:
            # Perform a Kolmogorov–Smirnov test for all combinations of results & connect the corresponding Violinplots
            for i in range(len(combinedNumbers)):
                dataX = df[args.metric][df['EvaluationModel'] ==
                                        os.path.basename(os.path.normpath(combinedFolders[i][0]))]
                dataY = df[args.metric][df['EvaluationModel'] ==
                                        os.path.basename(os.path.normpath(combinedFolders[i][1]))]

                KStest = ks_2samp(dataX, dataY)

                x1, x2 = combinedNumbers[i]

                y, h, col = df['dice_class0'].min() - 0.06 - 0.03 * i, -0.01, 'k'
                plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)

                # Show if the differentiation of the distributions is :
                # Not significant: ns, significant: *, very significant: ***
                if KStest.pvalue >= 0.5:
                    plt.text((x1 + x2) * .5, y + h, "ns", ha='center', va='bottom', color=col)
                elif 0.5 > KStest.pvalue >= 0.01:
                    plt.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color='r')
                elif KStest.pvalue < 0.01:
                    plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color='r')

        if args.metadata:
            plt.title("Metric:  " + args.metric + "\nMetadata:  "+ args.metadata[0] + ":" + args.metadata[1])
        else:
            plt.title("Metric:  " + args.metric)

        plt.grid()
        plt.show()

        print('success')

    else:
        print('No subjects meet the criteria selected for any model. '
              'Probably you need to change the --metadata / --metric selection')


if __name__ == "__main__":
    main(sys.argv[1:])

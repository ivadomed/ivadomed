import matplotlib
matplotlib.use('TkAgg') # This is needed for plotting through a CLI call
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import itertools
import seaborn as sns
from scipy.stats import ks_2samp
import argparse
import sys


# This function allows visualization of multiple evaluation models simultaneously and performs a Kolmogorov–Smirnov
# significance test between each combination of models.

# Konstantinos Nasiotis 2020


# Dependency: sudo apt-get install python3-tk
# For matplotlib visualization through a CLI call

# Example calls from terminal:
# python3 compare_testing_models.py --listfolders /home/nas/Desktop/logs/logs_NO_FILM_sctUsers /home/nas/Desktop/logs/logs_onlyT1w /home/nas/Desktop/logs/logs_onlyT2w
# or
# python3 compare_testing_models.py --listfolders /home/nas/Desktop/logs/logs_*
# ----------------------------------------------------------------------------------------------------------------------#

def main(argv):
    # defined command line options
    # this also generates --help and error handling
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--listfolders",
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=[],  # default if nothing is provided - This should give an error later on
    )

    # parse the command line
    args = CLI.parse_args()
    # access CLI options
    print("listfolders: %r" % args.listfolders)

    # Get the list
    logFoldersToCompare = args.listfolders

    if len(logFoldersToCompare) < 2:
        raise Exception('Less than two folders were selected - Nothing to compare')

    columnNames = ["EvaluationModel", 'dice_class0']
    df = pd.DataFrame([], columns=columnNames)

    for folder in logFoldersToCompare:
        result = pd.read_csv(os.path.join(folder, 'results_eval', 'evaluation_3Dmetrics.csv'))
        diceScores = result['dice_class0']
        folders = [os.path.basename(os.path.normpath(folder))] * len(diceScores)
        combined = np.column_stack((folders, diceScores.astype(np.object, folders))).T
        singleFolderDF = pd.DataFrame(combined, columnNames).T
        df = df.append(singleFolderDF, ignore_index=True)

    nFolders = len(logFoldersToCompare)
    combinedNumbers = list(itertools.combinations(range(nFolders), 2))
    combinedFolders = list(itertools.combinations(logFoldersToCompare, 2))

    # Pandas annoying issues
    df['dice_class0'] = df['dice_class0'].astype('float64')

    # Plot all violinplots
    sns.violinplot(x="EvaluationModel", y="dice_class0", data=df, color="0.8", inner='quartile')
    sns.stripplot(x="EvaluationModel", y="dice_class0", data=df, jitter=True, zorder=1)

    # Display the mean performance on top of every violinplot
    for i in range(len(logFoldersToCompare)):
        # This will be used to plot the mean value on top of each individual violinplot
        temp = df['dice_class0'][df['EvaluationModel'] == os.path.basename(os.path.normpath(logFoldersToCompare[i]))]
        plt.text(i, df['dice_class0'].max() + 0.07, str((100 * temp.mean()).round() / 100), ha='center', va='top',
                 color='r')

    # Perform a Kolmogorov–Smirnov test for every combination of results and connect the corresponding violinplots
    for i in range(len(combinedNumbers)):
        dataX = df['dice_class0'][df['EvaluationModel'] == os.path.basename(os.path.normpath(combinedFolders[i][0]))]
        dataY = df['dice_class0'][df['EvaluationModel'] == os.path.basename(os.path.normpath(combinedFolders[i][1]))]

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

    plt.grid()
    plt.show()

    print('success')


if __name__ == "__main__":
    main(sys.argv[1:])

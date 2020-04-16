#!/usr/bin/env python
##############################################################
#
# This script computes statistics to compare models
#
# Usage: python dev/compare_models.py -df path/to/dataframe.csv -n number_of_iterations --test-set
#
# Contributors: Olivier
#
##############################################################


import argparse
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind_from_stats


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--dataframe", required=True,
                        help="Path to saved dataframe (csv file).")
    parser.add_argument("-n", "--n-iterations", required=True, dest="n_iterations",
                        type=int, help="Number of times each config was run .")
    parser.add_argument("--test-set", dest='run_test', action='store_true',
                        help="Evaluate the trained model on the testing sub-set instead of validation.")
    return parser


def compute_statistics(results_df, n_iterations, run_test):

    avg = results_df.groupby(['log_directory']).mean()
    std = results_df.groupby(['log_directory']).std()

    print("Average dataframe")
    print(avg)
    print("Standard deviation dataframe")
    print(std)

    config_logs = list(avg.index.values)
    p_values = np.zeros((len(config_logs), len(config_logs)))
    i, j = 0, 0
    for confA in config_logs:
        j = 0
        for confB in config_logs:
            if run_test:
                p_values[i, j] = ttest_ind_from_stats(mean1=avg.loc[confA]["test_dice"], std1=std.loc[confA]["test_dice"],
                                                      nobs1=n_iterations, mean2=avg.loc[confB]["test_dice"], std2=std.loc[confB]["test_dice"], nobs2=n_iterations).pvalue
            else:
                p_values[i, j] = ttest_ind_from_stats(mean1=avg.loc[confA]["best_validation_dice"], std1=std.loc[confA]["best_validation_dice"],
                                                      nobs1=n_iterations, mean2=avg.loc[confB]["best_validation_dice"], std2=std.loc[confB]["best_validation_dice"], nobs2=n_iterations).pvalue
            j += 1
        i += 1

    p_df = pd.DataFrame(p_values, index=config_logs, columns=config_logs)
    print("P-values dataframe")
    print(p_df)


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    df = pd.read_csv(args.dataframe)
    compute_statistics(df, args.n_iterations, args.run_test)

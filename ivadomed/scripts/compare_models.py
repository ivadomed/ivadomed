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
from ivadomed import utils as imed_utils
from scipy.stats import ttest_ind_from_stats
from loguru import logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--dataframe", required=True,
                        help="Path to saved dataframe (csv file).",
                        metavar=imed_utils.Metavar.file)
    parser.add_argument("-n", "--n-iterations", required=True, dest="n_iterations",
                        type=int, help="Number of times each config was run .",
                        metavar=imed_utils.Metavar.int)
    parser.add_argument("--run_test", dest='run_test', action='store_true',
                        help="""Evaluate the trained model on the testing sub-set instead of
                                validation.""")
    parser.add_argument("-o", "--output", dest='out', default="comparison_models.csv",
                        help="if defined will represents the output csv file.",
                        metavar=imed_utils.Metavar.file)
    return parser


def compute_statistics(dataframe, n_iterations, run_test=True, csv_out='comparison_models.csv'):
    """Compares the performance of models at inference time on a common testing dataset using paired t-tests.

    It uses a dataframe generated by ``scripts/automate_training.py`` with the parameter ``--run-test`` (used to run the
    models on the testing dataset). It output dataframes that stores the different statistic (average, std and p_value
    between runs). All can be combined and stored in a csv.

    .. csv-table:: Example of dataframe
       :file: ../../images/df_compare.csv

    Usage example::

        ivadomed_compare_models -df results.csv -n 2 --run_test

    Args:
        dataframe (pandas.Dataframe): Dataframe of results generated by automate_training. Flag: ``--dataframe``, ``-df``
        n_iterations (int): Indicates the number of time that each experiment (ie set of parameter) was run.
                            Flag: ``--n_iteration``, ``-n``
        run_test (int): Indicates if the comparison is done on the performances on either the testing subdataset (True)
            either on the training/validation subdatasets. Flag: ``--run_test``
        csv_out (string): Output csv name to store computed value (e.g., df.csv). Default value is model_comparison.csv. Flag ``-o``, ``--output``
    """
    avg = dataframe.groupby(['path_output']).mean()
    std = dataframe.groupby(['path_output']).std()

    logger.info(f"Average dataframe: {avg}")
    logger.info(f"Standard deviation dataframe: {std}")

    config_logs = list(avg.index.values)
    p_values = np.zeros((len(config_logs), len(config_logs)))
    i, j = 0, 0
    for confA in config_logs:
        j = 0
        for confB in config_logs:
            if run_test:
                p_values[i, j] = ttest_ind_from_stats(mean1=avg.loc[confA]["test_dice"],
                                                      std1=std.loc[confA]["test_dice"],
                                                      nobs1=n_iterations, mean2=avg.loc[confB]["test_dice"],
                                                      std2=std.loc[confB]["test_dice"], nobs2=n_iterations).pvalue
            else:
                p_values[i, j] = ttest_ind_from_stats(mean1=avg.loc[confA]["best_validation_dice"],
                                                      std1=std.loc[confA]["best_validation_dice"],
                                                      nobs1=n_iterations, mean2=avg.loc[confB]["best_validation_dice"],
                                                      std2=std.loc[confB]["best_validation_dice"],
                                                      nobs2=n_iterations).pvalue
            j += 1
        i += 1

    p_df = pd.DataFrame(p_values, index=config_logs, columns=config_logs)
    logger.info("P-values dataframe")
    logger.info(p_df)
    if csv_out is not None:
        # Unnamed 0 column correspond to run number so we remoe that and add prefix for better readability
        df_concat = pd.concat([avg.add_prefix('avg_').drop(['avg_Unnamed: 0'], axis=1),
                               std.add_prefix('std_').drop(['std_Unnamed: 0'], axis=1), p_df.add_prefix('p-value_')],
                              axis=1)
        df_concat.to_csv(csv_out)


def main(args=None):
    imed_utils.init_ivadomed()
    parser = get_parser()
    args = imed_utils.get_arguments(parser, args)
    df = pd.read_csv(args.dataframe)
    compute_statistics(df, int(args.n_iterations), bool(args.run_test), args.out)


if __name__ == '__main__':
    main()

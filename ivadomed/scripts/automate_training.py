#!/usr/bin/env python
##############################################################
#
# This script enables training and comparison of models on multiple GPUs.
#
# Usage: python scripts/automate_training.py -c path/to/config.json -p path/to/hyperparams.json -n number_of_iterations --all-combin
#
##############################################################

import argparse
import copy
from functools import partial
import json
import logging
import os
import random
import shutil
import sys
from itertools import product

import joblib
import pandas as pd
import torch.multiprocessing as mp

from ivadomed import main as ivado
from ivadomed import config_manager as imed_config_manager
from ivadomed.utils import init_ivadomed
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.scripts.compare_models import compute_statistics

LOG_FILENAME = 'log.txt'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Base config file path.")
    parser.add_argument("-p", "--params", required=True, help="JSON file where hyperparameters to experiment are "
                                                              "listed.")
    parser.add_argument("-n", "--n-iterations", dest="n_iterations", default=1,
                        type=int, help="Number of times to run each config.")
    parser.add_argument("--all-combin", dest='all_combin', action='store_true',
                        help="To run all combinations of config")
    parser.add_argument("--run-test", dest='run_test', action='store_true',
                        help="Evaluate the trained model on the testing sub-set.")
    parser.add_argument("--fixed-split", dest='fixed_split', action='store_true',
                        help="Keep a constant dataset split for all configs and iterations")
    parser.add_argument("-l", "--all-logs", dest="all_logs", action='store_true',
                        help="Keep all log directories for each iteration.")
    parser.add_argument('-t', '--thr-increment', dest="thr_increment", required=False, type=float,
                        help="A threshold analysis is performed at the end of the training using the trained model and "
                             "the validation sub-dataset to find the optimal binarization threshold. The specified "
                             "value indicates the increment between 0 and 1 used during the analysis (e.g. 0.1).")

    return parser


def train_worker(config, thr_incr):
    current = mp.current_process()
    # ID of process used to assign a GPU
    ID = int(current.name[-1]) - 1

    # Use GPU i from the array specified in the config file
    config["gpu"] = config["gpu"][ID]

    # Call ivado cmd_train
    try:
        # Save best validation score
        best_training_dice, best_training_loss, best_validation_dice, best_validation_loss = \
            ivado.run_command(config, thr_increment=thr_incr)

    except:
        logging.exception('Got exception on main handler')
        print("Unexpected error:", sys.exc_info()[0])
        raise

    # Save config file in log directory
    config_copy = open(config["log_directory"] + "/config_file.json", "w")
    json.dump(config, config_copy, indent=4)

    return config["log_directory"], best_training_dice, best_training_loss, best_validation_dice, best_validation_loss


def test_worker(config):
    # Call ivado cmd_eval

    current = mp.current_process()
    # ID of process used to assign a GPU
    ID = int(current.name[-1]) - 1

    # Use GPU i from the array specified in the config file
    config["gpu"] = config["gpu"][ID]

    try:
        # Save best test score
        config["command"] = "test"
        df_results, test_dice = ivado.run_command(config)

    except:
        logging.exception('Got exception on main handler')
        print("Unexpected error:", sys.exc_info()[0])
        raise

    return config["log_directory"], test_dice, df_results


def make_category(base_item, keys, values, is_all_combin=False):
    items = []
    names = []

    if is_all_combin:
        for combination in product(*values):
            new_item = copy.deepcopy(base_item)
            name_str = ""
            for i in range(len(keys)):
                new_item[keys[i]] = combination[i]
                name_str += "-" + str(keys[i]) + "=" + str(combination[i]).replace("/", "_")

            items.append(new_item)
            names.append(name_str)

    else:
        for value_list, key in zip(values, keys):
            for value in value_list:
                new_item = copy.deepcopy(base_item)
                new_item[key] = value
                items.append(new_item)
                # replace / by _ to avoid creating new paths
                names.append("-" + str(key) + "=" + str(value).replace("/", "_"))

    return items, names


def automate_training(config, param, fixed_split, all_combin, n_iterations=1, run_test=False, all_logs=False,
                      thr_increment=None):
    """Automate multiple training processes on multiple GPUs.

    Hyperparameter optimization of models is tedious and time-consuming. This function automatizes this optimization
    across multiple GPUs. It runs trainings, on the same training and validation datasets, by combining a given set of
    parameters and set of values for each of these parameters. Results are collected for each combination and reported
    into a dataframe to allow their comparison. The script efficiently allocates each training to one of the available
    GPUs.

    Usage example::

        ivadomed_automate_training -c config.json -p params.json -n n_iterations

    .. csv-table:: Example of dataframe
       :file: ../../images/detailed_results.csv

    Args:
        config (string): Configuration filename, which is used as skeleton to configure the training. Some of its
            parameters (defined in `param` file) are modified across experiments. Flag: ``--config``, ``-c``
        param (string): json file containing parameters configurations to compare. Parameter "keys" of this file
            need to match the parameter "keys" of `config` file. Parameter "values" are in a list. Flag: ``--param``, ``-p``

            Example::

                {"default_model": {"depth": [2, 3, 4]}}

        fixed_split (bool): If True, all the experiments are run on the same training/validation/testing subdatasets.
                            Flag: ``--fixed-split``
        all_combin (bool): If True, all parameters combinations are run. Flag: ``--all-combin``
        n_iterations (int): Controls the number of time that each experiment (ie set of parameter) are run.
                            Flag: ``--n-iteration``, ``-n``
        run_test (bool): If True, the trained model is also run on the testing subdataset. flag: ``--run-test``
        all_logs (bool): If True, all the log directories are kept for every iteration. Flag: ``--all-logs``, ``-l``
        thr_increment (float): A threshold analysis is performed at the end of the training using the trained model and
            the validation sub-dataset to find the optimal binarization threshold. The specified value indicates the
            increment between 0 and 1 used during the ROC analysis (e.g. 0.1). Flag: ``-t``, ``--thr-increment``
    """
    # Load initial config
    initial_config = imed_config_manager.ConfigurationManager(config).get_config()

    # Hyperparameters values to experiment
    with open(param, "r") as fhandle:
        hyperparams = json.load(fhandle)
    param_dict, names_dict = {}, {}
    for category in hyperparams.keys():
        assert category in initial_config
        base_item = initial_config[category]
        keys = list(hyperparams[category].keys())
        values = [hyperparams[category][k] for k in keys]
        new_parameters, names = make_category(base_item, keys, values, all_combin)
        param_dict[category] = new_parameters
        names_dict[category] = names

    # Split dataset if not already done
    if fixed_split and (initial_config.get("split_path") is None):
        train_lst, valid_lst, test_lst = imed_loader_utils.get_new_subject_split(
            path_folder=initial_config["loader_parameters"]["bids_path"],
            center_test=initial_config["split_dataset"]["center_test"],
            split_method=initial_config["split_dataset"]["method"],
            random_seed=initial_config["split_dataset"]["random_seed"],
            train_frac=initial_config["split_dataset"]["train_fraction"],
            test_frac=initial_config["split_dataset"]["test_fraction"],
            log_directory="./",
            balance=initial_config["split_dataset"]['balance'] if 'balance' in initial_config["split_dataset"] else None)

        # save the subject distribution
        split_dct = {'train': train_lst, 'valid': valid_lst, 'test': test_lst}
        split_path = "./" + "common_split_datasets.joblib"
        joblib.dump(split_dct, split_path)
        initial_config["split_dataset"]["fname_split"] = split_path

    config_list = []
    # Test all combinations (change multiple parameters for each test)
    if all_combin:

        # Cartesian product (all combinations)
        combinations = (dict(zip(param_dict.keys(), values))
                        for values in product(*param_dict.values()))
        names = list(product(*names_dict.values()))

        for idx, combination in enumerate(combinations):

            new_config = copy.deepcopy(initial_config)

            for i, param in enumerate(combination):
                value = combination[param]
                new_config[param] = value
                new_config["log_directory"] = new_config["log_directory"] + names[idx][i]

            config_list.append(copy.deepcopy(new_config))
    # Change a single parameter for each test
    else:
        for param in param_dict:
            new_config = copy.deepcopy(initial_config)
            for value, name in zip(param_dict[param], names_dict[param]):
                new_config[param] = value
                new_config["log_directory"] = initial_config["log_directory"] + name
                config_list.append(copy.deepcopy(new_config))

    # CUDA problem when forking process
    # https://github.com/pytorch/pytorch/issues/2517
    mp.set_start_method('spawn')

    # Run all configs on a separate process, with a maximum of n_gpus  processes at a given time
    pool = mp.Pool(processes=len(initial_config["gpu"]))

    results_df = pd.DataFrame()
    eval_df = pd.DataFrame()
    all_mean = pd.DataFrame()
    for i in range(n_iterations):
        if not fixed_split:
            # Set seed for iteration
            seed = random.randint(1, 10001)
            for config in config_list:
                config["split_dataset"]["random_seed"] = seed
                if all_logs:
                    if i:
                        config["log_directory"] = config["log_directory"].replace("_n=" + str(i - 1).zfill(2),
                                                                                  "_n=" + str(i).zfill(2))
                    else:
                        config["log_directory"] += "_n=" + str(i).zfill(2)
        validation_scores = pool.map(partial(train_worker, thr_incr=thr_increment), config_list)
        val_df = pd.DataFrame(validation_scores, columns=[
            'log_directory', 'best_training_dice', 'best_training_loss', 'best_validation_dice',
            'best_validation_loss'])

        if run_test:
            new_config_list = []
            for config in config_list:
                # Delete path_pred
                path_pred = os.path.join(config['log_directory'], 'pred_masks')
                if os.path.isdir(path_pred) and n_iterations > 1:
                    try:
                        shutil.rmtree(path_pred)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

                # Take the config file within the log_directory because binarize_prediction may have been updated
                json_path = os.path.join(config['log_directory'], 'config_file.json')
                new_config = imed_config_manager.ConfigurationManager(json_path).get_config()
                new_config["gpu"] = config["gpu"]
                new_config_list.append(new_config)

            test_results = pool.map(test_worker, new_config_list)

            df_lst = []
            # Merge all eval df together to have a single excel file
            for j, result in enumerate(test_results):
                df = result[-1]

                if i == 0:
                    all_mean = df.mean(axis=0)
                    std_metrics = df.std(axis=0)
                    metrics = pd.concat([all_mean, std_metrics], sort=False, axis=1)
                else:
                    all_mean = pd.concat([all_mean, df.mean(axis=0)], sort=False, axis=1)
                    mean_metrics = all_mean.mean(axis=1)
                    std_metrics = all_mean.std(axis=1)
                    metrics = pd.concat([mean_metrics, std_metrics], sort=False, axis=1)

                metrics.rename({0: "mean"}, axis=1, inplace=True)
                metrics.rename({1: "std"}, axis=1, inplace=True)
                id = result[0].split("_n=")[0]
                cols = metrics.columns.values
                for idx, col in enumerate(cols):
                    metrics.rename({col: col + "_" + id}, axis=1, inplace=True)
                df_lst.append(metrics)
                test_results[j] = result[:2]

            # Init or add eval results to dataframe
            eval_df = pd.concat(df_lst, sort=False, axis=1)

            test_df = pd.DataFrame(test_results, columns=['log_directory', 'test_dice'])
            combined_df = val_df.set_index('log_directory').join(test_df.set_index('log_directory'))
            combined_df = combined_df.reset_index()

        else:
            combined_df = val_df

        results_df = pd.concat([results_df, combined_df])
        results_df.to_csv("temporary_results.csv")
        eval_df.to_csv("average_eval.csv")

    # Merge config and results in a df
    config_df = pd.DataFrame.from_dict(config_list)
    keep = list(param_dict.keys())
    keep.append("log_directory")
    config_df = config_df[keep]

    results_df = config_df.set_index('log_directory').join(results_df.set_index('log_directory'))
    results_df = results_df.reset_index()
    results_df = results_df.sort_values(by=['best_validation_loss'])

    results_df.to_csv("detailed_results.csv")

    print("Detailed results")
    print(results_df)

    # Compute avg, std, p-values
    if n_iterations > 1:
        compute_statistics(results_df, n_iterations, run_test)


def main():
    init_ivadomed()

    parser = get_parser()
    args = parser.parse_args()

    # Get thr increment if available
    thr_increment = args.thr_increment if args.thr_increment else None

    # Run automate training
    automate_training(args.config, args.params, bool(args.fixed_split), bool(args.all_combin), int(args.n_iterations),
                      bool(args.run_test), args.all_logs, thr_increment)


if __name__ == '__main__':
    main()

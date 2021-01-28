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
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.scripts.compare_models import compute_statistics
from ivadomed import utils as imed_utils

LOG_FILENAME = 'log.txt'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Base config file path.",
                        metavar=imed_utils.Metavar.file)
    parser.add_argument("-p", "--params", required=True,
                        help="JSON file where hyperparameters to experiment are listed.",
                        metavar=imed_utils.Metavar.file)
    parser.add_argument("-n", "--n-iterations", dest="n_iterations", default=1,
                        type=int, help="Number of times to run each config.",
                        metavar=imed_utils.Metavar.int)
    parser.add_argument("--all-combin", dest='all_combin', action='store_true',
                        help="To run all combinations of config"),
    parser.add_argument("-m", "--multi-params", dest="multi_params", action='store_true',
                        help="To change multiple parameters at once.")
    parser.add_argument("--run-test", dest='run_test', action='store_true',
                        help="Evaluate the trained model on the testing sub-set.")
    parser.add_argument("--fixed-split", dest='fixed_split', action='store_true',
                        help="Keep a constant dataset split for all configs and iterations")
    parser.add_argument("-l", "--all-logs", dest="all_logs", action='store_true',
                        help="Keep all log directories for each iteration.")
    parser.add_argument('-t', '--thr-increment', dest="thr_increment", required=False, type=float,
                        help="""A threshold analysis is performed at the end of the training using
                                the trained model and the validation sub-dataset to find the optimal
                                binarization threshold. The specified value indicates the increment
                                between 0 and 1 used during the analysis (e.g. 0.1).""",
                        metavar=imed_utils.Metavar.float)
    parser.add_argument("-o", "--output_dir", required=False,
                        help="Output Folder.")

    return parser


def train_worker(config, thr_incr):
    """
    Args:
        config (dict): dictionary containing configuration details.
        thr_incr (float): A threshold analysis is performed at the end of the training
            using the trained model and the validation sub-dataset to find the optimal binarization
            threshold. The specified value indicates the increment between 0 and 1 used during the
            ROC analysis (e.g. 0.1). Flag: ``-t``, ``--thr-increment``
    """
    current = mp.current_process()
    # ID of process used to assign a GPU
    ID = int(current.name[-1]) - 1

    # Use GPU i from the array specified in the config file
    config["gpu_ids"] = [config["gpu_ids"][ID]]

    # Call ivado cmd_train
    try:
        # Save best validation score
        config["command"] = "train"
        best_training_dice, best_training_loss, best_validation_dice, best_validation_loss = \
            ivado.run_command(config, thr_increment=thr_incr)

    except Exception:
        logging.exception('Got exception on main handler')
        logging.info("Unexpected error:", sys.exc_info()[0])
        raise

    # Save config file in output path
    config_copy = open(config["path_output"] + "/config_file.json", "w")
    json.dump(config, config_copy, indent=4)

    return config["path_output"], best_training_dice, best_training_loss, best_validation_dice, \
        best_validation_loss


def test_worker(config):
    # Call ivado cmd_eval

    current = mp.current_process()
    # ID of process used to assign a GPU
    ID = int(current.name[-1]) - 1

    # Use GPU i from the array specified in the config file
    config["gpu_ids"] = [config["gpu_ids"][ID]]

    try:
        # Save best test score
        config["command"] = "test"
        df_results, test_dice = ivado.run_command(config)

    except Exception:
        logging.exception('Got exception on main handler')
        logging.info("Unexpected error:", sys.exc_info()[0])
        raise

    return config["path_output"], test_dice, df_results


def make_category(base_item, keys, values, is_all_combin=False, multiple_params=False):
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
    elif multiple_params:
        value_len = set()
        for value in values:
            value_len.add(len(value))
        if len(value_len) != 1:
            raise ValueError("To use flag --multi-params or -m, all hyperparameter lists need to be the same size.")

        for v_idx in range(len(values[0])):
            name_str = ""
            new_item = copy.deepcopy(base_item)
            for k_idx, key in enumerate(keys):
                new_item[key] = values[k_idx][v_idx]
                name_str += "-" + str(key) + "=" + str(values[k_idx][v_idx]).replace("/", "_")

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


def automate_training(config, param, fixed_split, all_combin, n_iterations=1, run_test=False,
                      all_logs=False, thr_increment=None, multiple_params=False,
                      output_dir=None):
    """Automate multiple training processes on multiple GPUs.

    Hyperparameter optimization of models is tedious and time-consuming.
    This function automatizes this optimization across multiple GPUs. It runs trainings, on the
    same training and validation datasets, by combining a given set of parameters and set of
    values for each of these parameters. Results are collected for each combination and reported
    into a dataframe to allow their comparison. The script efficiently allocates each training to
    one of the available GPUs.

    Usage example::

        ivadomed_automate_training -c config.json -p params.json -n n_iterations

    .. csv-table:: Example of dataframe
       :file: ../../images/detailed_results.csv

    Args:
        config (string): Configuration filename, which is used as skeleton to configure the
            training. Some of its parameters (defined in `param` file) are modified across
            experiments. Flag: ``--config``, ``-c``
        param (string): json file containing parameters configurations to compare.
            Parameter "keys" of this file need to match the parameter "keys" of `config` file.
            Parameter "values" are in a list. Flag: ``--param``, ``-p``

            Example::

                {"default_model": {"depth": [2, 3, 4]}}

        fixed_split (bool): If True, all the experiments are run on the same
            training/validation/testing subdatasets. Flag: ``--fixed-split``
        all_combin (bool): If True, all parameters combinations are run. Flag: ``--all-combin``
        n_iterations (int): Controls the number of time that each experiment (ie set of parameter)
            are run. Flag: ``--n-iteration``, ``-n``
        run_test (bool): If True, the trained model is also run on the testing subdataset.
            Flag: ``--run-test``
        all_logs (bool): If True, all the log directories are kept for every iteration.
            Flag: ``--all-logs``, ``-l``
        thr_increment (float): A threshold analysis is performed at the end of the training
            using the trained model and the validation sub-dataset to find the optimal binarization
            threshold. The specified value indicates the increment between 0 and 1 used during the
            ROC analysis (e.g. 0.1). Flag: ``-t``, ``--thr-increment``
        multiple_params (bool): If True, more than one parameter will be change at the time from
            the hyperparameters. All the first elements from the hyperparameters list will be
            applied, then all the second, etc.
        output_dir (str): Path to where the results will be saved.
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
        new_parameters, names = make_category(base_item, keys, values, all_combin, multiple_params)
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
            path_output="./",
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
                new_config["path_output"] = new_config["path_output"] + names[idx][i]

            config_list.append(copy.deepcopy(new_config))
    elif multiple_params:
        for config_idx in range(len(names)):
            new_config = copy.deepcopy(initial_config)
            config_name = ""
            for param in param_dict:
                new_config[param] = param_dict[param][config_idx]
                config_name += names_dict[param][config_idx]
            new_config["path_output"] = initial_config["path_output"] + config_name
            config_list.append(copy.deepcopy(new_config))

    # Change a single parameter for each test
    else:
        for param in param_dict:
            new_config = copy.deepcopy(initial_config)
            for value, name in zip(param_dict[param], names_dict[param]):
                new_config[param] = value
                new_config["path_output"] = initial_config["path_output"] + name
                config_list.append(copy.deepcopy(new_config))

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not output_dir:
        output_dir = ""

    # CUDA problem when forking process
    # https://github.com/pytorch/pytorch/issues/2517
    ctx = mp.get_context("spawn")

    # Run all configs on a separate process, with a maximum of n_gpus  processes at a given time
    logging.info(initial_config['gpu_ids'])

    results_df = pd.DataFrame()
    eval_df = pd.DataFrame()
    all_mean = pd.DataFrame()

    with ctx.Pool(processes=len(initial_config["gpu_ids"])) as pool:
        for i in range(n_iterations):
            if not fixed_split:
                # Set seed for iteration
                seed = random.randint(1, 10001)
                for config in config_list:
                    config["split_dataset"]["random_seed"] = seed
                    if all_logs:
                        if i:
                            config["path_output"] = config["path_output"].replace("_n=" + str(i - 1).zfill(2),
                                                                                      "_n=" + str(i).zfill(2))
                        else:
                            config["path_output"] += "_n=" + str(i).zfill(2)

                validation_scores = pool.map(partial(train_worker, thr_incr=thr_increment), config_list)

            val_df = pd.DataFrame(validation_scores, columns=[
                'path_output', 'best_training_dice', 'best_training_loss', 'best_validation_dice',
                'best_validation_loss'])

            if run_test:
                new_config_list = []
                for config in config_list:
                    # Delete path_pred
                    path_pred = os.path.join(config['path_output'], 'pred_masks')
                    if os.path.isdir(path_pred) and n_iterations > 1:
                        try:
                            shutil.rmtree(path_pred)
                        except OSError as e:
                            logging.info("Error: %s - %s." % (e.filename, e.strerror))

                    # Take the config file within the path_output because binarize_prediction may have been updated
                    json_path = os.path.join(config['path_output'], 'config_file.json')
                    new_config = imed_config_manager.ConfigurationManager(json_path).get_config()
                    new_config["gpu_ids"] = config["gpu_ids"]
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

                test_df = pd.DataFrame(test_results, columns=['path_output', 'test_dice'])
                combined_df = val_df.set_index('path_output').join(test_df.set_index('path_output'))
                combined_df = combined_df.reset_index()

            else:
                combined_df = val_df

            results_df = pd.concat([results_df, combined_df])
            results_df.to_csv(os.path.join(output_dir, "temporary_results.csv"))
            eval_df.to_csv(os.path.join(output_dir, "average_eval.csv"))

    # Merge config and results in a df
    config_df = pd.DataFrame.from_dict(config_list)
    keep = list(param_dict.keys())
    keep.append("path_output")
    config_df = config_df[keep]

    results_df = config_df.set_index('path_output').join(results_df.set_index('path_output'))
    results_df = results_df.reset_index()
    results_df = results_df.sort_values(by=['best_validation_loss'])

    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"))

    logging.info("Detailed results")
    logging.info(results_df)

    # Compute avg, std, p-values
    if n_iterations > 1:
        compute_statistics(results_df, n_iterations, run_test)


def main(args=None):
    imed_utils.init_ivadomed()
    parser = get_parser()
    args = imed_utils.get_arguments(parser, args)

    # Get thr increment if available
    thr_increment = args.thr_increment if args.thr_increment else None

    automate_training(config=args.config,
                      param=args.params,
                      fixed_split=bool(args.fixed_split),
                      all_combin=bool(args.all_combin),
                      n_iterations=int(args.n_iterations),
                      run_test=bool(args.run_test),
                      all_logs=args.all_logs,
                      thr_increment=thr_increment,
                      multiple_params=bool(args.multi_params),
                      output_dir=args.output_dir
                      )


if __name__ == '__main__':
    main()

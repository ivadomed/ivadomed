#!/usr/bin/env python
##############################################################
#
# This script enables training and comparison of models on multiple GPUs
#
# Usage: python dev/automate_training.py -c path/to/config.json -n number_of_iterations --all-combin
#
# Contributors: Olivier
#
##############################################################

import argparse
import copy
import joblib
import json
import logging
import os
import pandas as pd
import random
import shutil
import sys
import torch.multiprocessing as mp

from ivadomed import main as ivado
from ivadomed.loader import utils as imed_loader_utils
from itertools import product
from dev.compare_models import compute_statistics

LOG_FILENAME = 'log.txt'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Base config file path.")
    parser.add_argument("-n", "--n-iterations", dest="n_iterations",
                        type=int, help="Number of times to run each config .")
    parser.add_argument("--all-combin", dest='all_combin', action='store_true',
                        help="To run all combinations of config")
    parser.add_argument("--run-test", dest='run_test', action='store_true',
                        help="Evaluate the trained model on the testing sub-set.")
    parser.add_argument("--fixed-split", dest='fixed_split', action='store_true',
                        help="Keep a constant dataset split for all configs and iterations")
    parser.set_defaults(all_combin=False)

    return parser


def train_worker(config):
    current = mp.current_process()
    # ID of process used to assign a GPU
    ID = int(current.name[-1]) - 1

    # Use GPU i from the array specified in the config file
    config["gpu"] = config["gpu"][ID]

    # Call ivado cmd_train
    try:
        # Save best validation score
        best_training_dice, best_training_loss, best_validation_dice, best_validation_loss = ivado.cmd_train(
            config)

    except:
        logging.exception('Got exception on main handler')
        print("Unexpected error:", sys.exc_info()[0])
        raise

    # Save config file in log directory
    config_copy = open(config["log_directory"] + "/config.json", "w")
    json.dump(config, config_copy, indent=4)

    return config["log_directory"], best_training_dice, best_training_loss, best_validation_dice, best_validation_loss


def test_worker(config):
    current = mp.current_process()
    # ID of process used to assign a GPU
    ID = int(current.name[-1]) - 1

    # Use GPU i from the array specified in the config file
    config["gpu"] = config["gpu"][ID]

    # Call ivado cmd_eval
    try:
        # Save best test score

        config["command"] = "eval"
        test_dict, eval_df = ivado.cmd_eval(config)
        test_dice = test_dict['dice_score']

        # Uncomment to use 3D dice
        # test_dice = eval_df["dice"].mean()

    except:
        logging.exception('Got exception on main handler')
        print("Unexpected error:", sys.exc_info()[0])
        raise

    return config["log_directory"], test_dice

def make_category(base_item, keys, values):
    items = []
    for combination in product(*values):
        new_item = copy.deepcopy(base_item)
        for i in range(len(keys)):
            new_item[keys[i]] = combination[i]

        items.append(new_item)
    return items

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    # Load initial config
    with open(args.config, "r") as fhandle:
        initial_config = json.load(fhandle)

    # Hyperparameters values to test

    # Step 1 : batch size, initial LR and LR scheduler

    ### Training parameters
    category = "training_parameters"
    base_item = initial_config[category]
    keys = ["batch_sizes", "initial_lrs", "lr_scheduler", "loss", "mixup_alpha"]

    batch_sizes = [8, 16, 32, 64]
    initial_lrs = [1e-2, 1e-3, 1e-4, 1e-5]
    lr_schedulers = [{"name": "CosineAnnealingLR"},
                    {"name": "CosineAnnealingWarmRestarts", "T_0": 10}
                    {"name": "CyclicLR", "base_lr" : X, "max_lr" : Y}]

    values = [batch_sizes, initial_lrs, lr_schedulers]

    #Losses
    ### Simple case (one config per loss type)
    losses = [{"name": "DiceLoss"},
            {"name": "cross_entropy"},
            {"name": "GeneralizedDiceLoss"},
            {"name": "FocalLoss", "params": {"gamma": 0.5, "alpha" : 0.2}}]

    ### Complex case (nested combinations)

    """
    base_loss = {"name": "focal", "params": {"gamma": 0.5, "alpha" : 0.2}}
    alphas = [0.2, 0.5, 0.75, 1]
    gammas = [0.5, 1, 1.5, 2]

    loss_params = make_category(initial_config["training_parameters"]["loss"]["params"], ["gamma","alpha"], [alphas, gammas])
    losses = make_category(initial_config["training_parameters"]["loss"], ["params"], [loss_params])
    """

    #MixUp
    mixup_alphas = [0.5, 1, 2]

    values = [batch_sizes, initial_lrs, lr_schedulers, losses, mixup_alphas]
    training_parameters = make_category(base_item, keys, values)


    # Step 2 : FiLM

    category = "FiLMedUnet"
    base_item = initial_config[category]
    keys = ["applied", "metadata", "film_layers"]

    applied = ["true"]
    metadata = ["contrasts"]

    film_layers = [ [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]]

    values = [applied, metadata, film_layers]
    film_parameters = make_category(base_item, keys, values)


    # Add other steps here

    # Split dataset if not already done

    if args.fixed_split and (initial_config.get("split_path") is None):
        train_lst, valid_lst, test_lst = imed_loader_utils.split_dataset(path_folder=initial_config["bids_path"],
                                                                         center_test_lst=initial_config["center_test"],
                                                                         split_method=initial_config["split_method"],
                                                                         random_seed=initial_config["random_seed"],
                                                                         train_frac=initial_config["train_fraction"],
                                                                         test_frac=initial_config["test_fraction"])

        # save the subject distribution
        split_dct = {'train': train_lst, 'valid': valid_lst, 'test': test_lst}
        split_path = "./" + "common_split_datasets.joblib"
        joblib.dump(split_dct, split_path)
        initial_config["split_path"] = split_path

    # Dict with key corresponding to name of the param in the config file
    param_dict = {"training_parameters": training_parameters, "FiLMedUnet": film_parameters}

    config_list = []
    # Test all combinations (change multiple parameters for each test)
    if args.all_combin:

        # Cartesian product (all combinations)
        combinations = (dict(zip(param_dict.keys(), values))
                        for values in product(*param_dict.values()))

        for combination in combinations:

            new_config = copy.deepcopy(initial_config)

            for param in combination:
                value = combination[param]
                new_config[param] = value
                new_config["log_directory"] = new_config["log_directory"] + "-" + param + "=" + str(value)

            config_list.append(copy.deepcopy(new_config))
    # Change a single parameter for each test
    else:
        for param in param_dict:

            new_config = copy.deepcopy(initial_config)

            for value in param_dict[param]:
                new_config[param] = value
                new_config["log_directory"] = initial_config["log_directory"] + "-" + param + "=" + str(value)
                config_list.append(copy.deepcopy(new_config))

    # CUDA problem when forking process
    # https://github.com/pytorch/pytorch/issues/2517
    mp.set_start_method('spawn')

    # Run all configs on a separate process, with a maximum of n_gpus  processes at a given time
    pool = mp.Pool(processes=len(initial_config["gpu"]))
    if args.n_iterations is not None:
        n_iterations = args.n_iterations
    else:
        n_iterations = 1

    results_df = pd.DataFrame()
    for i in range(n_iterations):
        if not args.fixed_split:
            # Set seed for iteration
            seed = random.randint(1, 10001)
            for config in config_list:
                config["random_seed"] = seed

        validation_scores = pool.map(train_worker, config_list)
        val_df = pd.DataFrame(validation_scores, columns=[
            'log_directory', 'best_training_dice', 'best_training_loss', 'best_validation_dice',
            'best_validation_loss'])

        if args.run_test:
            for config in config_list:
                # Delete path_pred
                path_pred = os.path.join(config['log_directory'], 'pred_masks')
                if os.path.isdir(path_pred) and n_iterations > 1:
                    try:
                        shutil.rmtree(path_pred)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

            test_scores = pool.map(test_worker, config_list)
            test_df = pd.DataFrame(test_scores, columns=['log_directory', 'test_dice'])
            combined_df = val_df.set_index('log_directory').join(
                test_df.set_index('log_directory'))
            combined_df = combined_df.reset_index()

        else:
            combined_df = val_df

        results_df = pd.concat([results_df, combined_df])
        results_df.to_csv("temporary_results.csv")
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
        compute_statistics(results_df, n_iterations, args.run_test)

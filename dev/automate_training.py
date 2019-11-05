##############################################################
#
# This script automates the training  of a networks on multiple GPUs to deal with hyperparameter optimisation
#
# Usage: python dev/training_scheduler.py -c path/to/config.json -g number_of_gpus
#
# Contributors: olivier
# Last modified: 5-11-2019
#
##############################################################

import argparse
import copy
import json
import logging
import pandas as pd
import sys
import torch.multiprocessing as mp
#import time

from ivadomed import main as ivado
from itertools import product

LOG_FILENAME = 'log.txt'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Base config file path.")
    parser.add_argument("--all-combin", dest='all_combin', action='store_true', help="To run all combinations of config")
    parser.add_argument("-g", "--n-gpus", type=int, dest='n_gpus', required=True, help="Number of GPUs to use")
    parser.set_defaults(all_combin=False)
    return parser

def worker(config):
    current = mp.current_process()
    #ID of process used to assign a GPU
    ID = current.name[-1]
    #Offset because Lucas uses GPU 0,1
    config["gpu"] =  int(ID) + 1
    #print(config["gpu"])

    #Call ivado cmd_train
    try:
        #Save best validation score
        best_val = ivado.cmd_train(config)
    except:
        logging.exception('Got exception on main handler')
        print("Unexpected error:", sys.exc_info()[0])
        raise

    # Save config file in log directory
    config_copy = open(config["log_directory"] + "/config.json","w")
    json.dump(config, config_copy, indent=4)

    return config["log_directory"],best_val


if __name__ == '__main__':

    #if len(sys.argv) <= 1:
    #    print("\n training_scheduler [initial_config.json]\n")
    #    exit()

    parser = get_parser()
    args = parser.parse_args()
    #Load initial config
    with open(args.config, "r") as fhandle:
        initial_config = json.load(fhandle)

    #Number of GPUs we want to use
    n_gpus = args.n_gpus

    #Parameters to test
    batch_sizes = [8, 16, 32, 64]
    initial_lrs = [1e-2, 1e-3, 1e-4, 1e-5]

    #Other parameters
    #gt_dilations = [0, 0.5, 1]
    #metadatas = ["without", "contrast", "mri_params"]
    #film_layers = [ [1, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 1, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 1],
    #                [1, 1, 1, 1, 1, 1, 1, 1]]
    #mixup_bools = [False, True]
    #mixup_alphas = [2]
    #losses = [{"name": "dice"}, {"name": "focal", "params": {"gamma": 0.5}}]
    #lr_schedulers = [{"name": "CosineAnnealingWarmRestarts", "T_0": 10}]


    #Dict with key corresponding to name of the param in the config file
    param_dict = {"batch_size":batch_sizes, "initial_lr":initial_lrs}

    config_list = []
    if args.all_combin:

        #Cartesian product (all combinations)
        combinations = (dict(zip(param_dict.keys(), values)) for values in product(*param_dict.values()))

        for combination in combinations:

            #Change multiple parameters for each config
            new_config = copy.deepcopy(initial_config)

            for param in combination:

                value = combination[param]
                new_config[param] = value
                new_config["log_directory"] = new_config["log_directory"] + "-" + param + "=" + str(value)

            config_list.append(copy.deepcopy(new_config))
    else:
        for param in param_dict:

            #Change only one parameter for each config
            new_config = copy.deepcopy(initial_config)

            for value in param_dict[param]:

                new_config[param] = value
                new_config["log_directory"] = initial_config["log_directory"] + "-" + param + "=" + str(value)
                config_list.append(copy.deepcopy(new_config))

    #for el in config_list:
        #print(el["log_directory"])
        #print(el)

    #exit()

    #CUDA problem when forking process
    #https://github.com/pytorch/pytorch/issues/2517
    mp.set_start_method('spawn')

    #Run all configs on a separate process, with a maximum of n_gpus  processes at a given time
    pool = mp.Pool(processes = n_gpus)
    best_val = pool.map(worker,config_list)


    #Merge config and results in a df
    config_df = pd.DataFrame.from_dict(config_list)
    keep = list(param_dict.keys())
    keep.append("log_directory")
    config_df = config_df[keep]

    results_df = pd.DataFrame(best_val, columns =['log_directory', 'best_val'])
    results_df = config_df.set_index('log_directory').join(results_df.set_index('log_directory'))
    results_df = results_df.sort_values(by=['best_val'])

    results_df.to_pickle(output_df.pkl)
    print(results_df.drop(["log_directory"]))

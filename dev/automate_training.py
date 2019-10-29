##############################################################
#
# This script automates the training  of a networks on multiple GPUs to deal with hyperparameter optimisation
#
# Usage: python dev/training_scheduler.py config/config.json
#
# Contributors: olivier
# Last modified: 29-10-2019
#
##############################################################

import sys
import json
#import torch
import torch.multiprocessing as mp
#from torch.multiprocessing.pool import ThreadPool
import copy
import time
from ivadomed import main as ivado
import logging

LOG_FILENAME = 'log.out'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


def worker(config):
    current = mp.current_process()
    #ID of process (can be assigned to a GPU)
    ID = current.name[-1]
    #Offset because Lucas uses GPU 0,1
    config["gpu"] =  int(ID) + 1
    print(config["gpu"])
    #time.sleep(2)

    #Call ivado cmd_train
    try:
        ivado.cmd_train(config)
    except:
        logging.exception('Got exception on main handler')
        print("Unexpected error:", sys.exc_info()[0])
        raise

    print("finish")

    #Todo save config file in log dir

    return


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("\n training_scheduler [initial_config.json]\n")
        exit()

    #Load initial config
    with open(sys.argv[1], "r") as fhandle:
        initial_config = json.load(fhandle)

    #Number of GPUs we want to use
    n_gpus = 1

    #print(torch.cuda.is_available())
    #Parameters to test
    batch_sizes = [8, 16, 32, 64]
    initial_lrs = [1e-2, 1e-3, 1e-4, 1e-5]

    #Dict with key corresponding to name of the param in the config file
    param_dict = {"batch_size":batch_sizes, "initial_lr":initial_lrs}

    config_list = []
    for param in param_dict:

        #Change only one parameter at a time
        new_config = copy.deepcopy(initial_config)

        for value in param_dict[param]:

            new_config[param] = value
            new_config["log_directory"] = initial_config["log_directory"] + "_" + param + "_" + str(value)
            config_list.append(copy.deepcopy(new_config))

    #Test
    #print(len(config_list))
    #for el in config_list:
    #    print(el)

    #CUDA problem when forking process
    #https://github.com/pytorch/pytorch/issues/2517
    mp.set_start_method('spawn')

    #Run all configs on a separate process, with a maximum of n_gpus  processes at a given time
    pool = mp.Pool(processes = n_gpus)
    pool.map(worker,config_list)

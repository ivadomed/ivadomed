##############################################################
#
# This script automates the training  of a networks on multiple GPUs to deal with hyperparameter optimisation
#
# Usage: python dev/training_scheduler.py base_config.json
#
# Contributors: olivier
# Last modified: 29-10-2019
#
##############################################################

import sys
import json
import torch.multiprocessing as mp
import copy
import time
def f(config):
    current = mp.current_process()
    #ID of process (can be assigned to a GPU)
    ID = current.name[-1]
    config["gpu"] =  int(ID)
    print(config["gpu"])
    time.sleep(2)

    #Todo : call ivadomed with the modified config

    
    return


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("\n training_scheduler [initial_config.json]\n")
        exit()

    #Load initial config
    with open(sys.argv[1], "r") as fhandle:
        initial_config = json.load(fhandle)

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

    pool = mp.Pool(processes = 4)
    pool.map(f,config_list)

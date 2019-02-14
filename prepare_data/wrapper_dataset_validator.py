#!/usr/bin/env python
#
# Wrapper to check BIDS datasets within a root
#
# Usage:
#   python wrapper_check_dataset.py -r 
#
# Authors: Alexandru Foias, Julien Cohen-Adad

import os, argparse
import dataset_validator

def get_parameters():
    parser = argparse.ArgumentParser(description='Wrapper to check BIDS datasets within a root')
    parser.add_argument('-r', '--path-root-data',
                        help='Path to input root BIDS datasets directory.',
                        required=True)
    args = parser.parse_args()
    return args


def wrapper_dataset_validator(path_root_data):
    
    """
    Wrapper to check BIDS datasets within a root
    :param path_root_data: Path to input root BIDS datasets directory
    :return:
    """
    #list all directories within the root folder
    list_BIDS_dataset =  os.listdir(path_root_data)
    #loop accros the individual BIDS datasets
    for item in list_BIDS_dataset:
        path_bids_dataset = os.path.join(path_root_data,item)
        print '\n' + path_bids_dataset
        if os.path.isdir(path_bids_dataset):
            #call dataset_validator.py
            dataset_validator.check_bids_dataset(path_bids_dataset)

if __name__ == "__main__":
    args = get_parameters()
    wrapper_dataset_validator(args.path_root_data)
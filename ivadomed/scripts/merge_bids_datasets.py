from distutils.dir_util import copy_tree
import pandas as pd
import json
import glob
import os
import argparse
import sys
from ivadomed import utils as imed_utils


# This scripts merges several BIDS datasets.
# the new participants.tsv and participants.json are merged versions of the initial files.
# 2 Inputs should be added:
# 1. --ifolders: list of the Datasets / BIDS Folders to be merged
# 2. --ofolder: output folder

# Example call:
# python3 merge_merge_bids_datasets.py --ifolders ~/first_Dataset/ ~/second_Dataset/ --ofolder ~/mergedDataset/

# Konstantinos Nasiotis 2021

# -----------------------------------------------------------------------------------------------------------------------#


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ifolders", required=True,
                        help="Input BIDS folders.", nargs="*",
                        metavar=imed_utils.Metavar.file)
    parser.add_argument("-o", "--ofolder", required=False, default="./merged_bids_datasets",
                        help="Output folder for merged datasets.",
                        metavar=imed_utils.Metavar.file)
    parser.add_argument("-m", "--metadata", required=False, default=[],
                        help="Select files within a dataset that match specific metadata.",
                        metavar=imed_utils.Metavar.file)
    parser.add_argument("--copyfiles", required=False, default=False,
                        help="Copy the files from the original folders to the output folder.",
                        metavar=imed_utils.Metavar.file)
    return parser

'''
"--ifolders",
        nargs='*',  # Any number of folders expected to be merged - Must be higher than 1 dataset
        type=str,
        default=[],  # default if nothing is provided - This should give an error later on
    )
'''

def run_merging(input_folders, output_folder, metadata=[], copyfiles=False):

    # access CLI options
    print("Input folders: %r" % input_folders)
    print("Output folder: %r" % output_folder)


    print('Make sure there were no inconsistencies in column labels between the initial participants.tsv files '
          '- e.g. subject_id - subject_ids etc.')

    # Create output folder if it doesnt exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Load and merge the .tsv and .json files
    if isinstance(input_folders, str):
        raise TypeError("'bids_path' in the config file should be a list")
    elif len(input_folders) < 2:
        raise Exception("Less than 2 datasets selected. No need to merge")
    else:
        # Merge multiple .tsv files into the same dataframe
        #df_merged = pd.read_table(os.path.join(input_folders[0], 'participants.tsv'), encoding="ISO-8859-1")
        df_merged = pd.read_table(os.path.join(input_folders[0], 'participants.tsv'))

        # Convert to string to get rid of potential TypeError during merging within the same column
        df_merged = df_merged.astype(str)

        # Get first .json file
        jsonFile = os.path.join(input_folders[0], 'participants.json')
        with open(jsonFile) as json_file:
            json_merged = json.load(json_file)

        for iFolder in range(1, len(input_folders)):
            #df_next = pd.read_table(os.path.join(input_folders[iFolder], 'participants.tsv'), encoding="ISO-8859-1")
            df_next = pd.read_table(os.path.join(input_folders[iFolder], 'participants.tsv'), encoding="ISO-8859-1")
            df_next = df_next.astype(str)
            # Merge the .tsv files (This keeps also non-overlapping fields)
            df_merged = pd.merge(left=df_merged, right=df_next, how='outer')

            jsonFile_next = os.path.join(input_folders[iFolder], 'participants.json')
            with open(jsonFile_next) as json_file:
                jsonNext = json.load(json_file)

            # Merge .json files
            json_merged = {**json_merged, **jsonNext}

    # Create new .tsv and .json files from the merged entries
    df_merged.to_csv(os.path.join(output_folder, 'participants.tsv'), sep=',', index=False)

    with open(os.path.join(output_folder, 'participants.json'), 'w') as outfile:
        json.dump(json_merged, outfile, indent=4)

    # Create a dataset_decription.json -  This is needed on the BIDS loader
    # Confirm the version is correct
    with open(os.path.join(output_folder, 'dataset_description.json'), 'w') as outfile:
        json.dump({"BIDSVersion": "1.0.1", "Name": "Ivadomed BIDS Merger"}, outfile, indent=4)

    if copyfiles:
        # Start copying folders
        for datasetFolder in input_folders:
            subjectsFolders = glob.glob(os.path.join(datasetFolder, 'sub-*'))
            derivativesFolder = glob.glob(os.path.join(datasetFolder, 'derivatives'))

            if derivativesFolder != []:
                subjectsFolders.append(derivativesFolder[0])
                foldersToCopy = subjectsFolders
            else:
                foldersToCopy = subjectsFolders
                print("No derivatives are present in this folder")

            for subFolder in foldersToCopy:
                copy_tree(subFolder, os.path.join(output_folder, os.path.basename(os.path.normpath(subFolder))))


def main(args=None):
    imed_utils.init_ivadomed()
    parser = get_parser()
    args = imed_utils.get_arguments(parser, args)
    run_merging(input_folders=args.ifolders, output_folder=args.ofolder, metadata=args.metadata,
                      copyfiles=args.copyfiles)


if __name__ == '__main__':
    main()

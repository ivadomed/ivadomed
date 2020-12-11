from distutils.dir_util import copy_tree
from bids_neuropoly import bids
import pandas as pd
import json
import glob
import os
import argparse
import sys

# This scripts merges 2 BIDS datasets.
# the new participants.tsv and participants.json are merged versions of the initial files.
# 2 Inputs should be added:
# 1. --ifolders: list of the 2 Folders to be merged
# 2. --ofolder: output folder

# Example call:
# python3 merge_participants_files.py --ifolders ~/first_Dataset/ ~/second_Dataset/ --ofolder ~/mergedDataset/

# Konstantinos Nasiotis 2020

# -----------------------------------------------------------------------------------------------------------------------#


def main_run(argv):

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--ifolders",
        nargs=2,  # 2 folders expected to be merged
        type=str,
        default=[],  # default if nothing is provided - This should give an error later on
    )
    CLI.add_argument(
        "--ofolder",  # name on the CLI - drop the `--` for positional/required parameters
        nargs=1,  # 1 folder expected
        type=str,
        default=[],  # default if nothing is provided
    )

    # parse the command line
    args = CLI.parse_args()
    # access CLI options
    print("Input folders: %r" % args.ifolders)
    print("Output folder: %r" % args.ofolder)



    datasetFolder1 = args.ifolders[0]
    datasetFolder2 = args.ifolders[1]
    output_folder = args.ofolder[0]


    print('Make sure there were no inconsistencies in column labels between the two initial participants.tsv files - e.g. subject_id - subject_ids etc.')

    # Create output folder if it doesnt exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Load the .tsv files
    df1 = bids.BIDS(datasetFolder1).participants.content
    df2 = bids.BIDS(datasetFolder2).participants.content

    # Merge the .tsv files and save them in a new file (This keeps also non-overlapping fields)
    df_merged = pd.merge(left=df1, right=df2, how='outer')
    df_merged.to_csv(os.path.join(output_folder, 'participants.tsv'), sep='\t', index=False)


    # Do the same for the .json files
    jsonFile1 = os.path.join(datasetFolder1, 'participants.json')
    jsonFile2 = os.path.join(datasetFolder2, 'participants.json')

    with open(jsonFile1) as json_file:
        json1 = json.load(json_file)
    with open(jsonFile2) as json_file:
        json2 = json.load(json_file)

    # Merge .json files
    json_merged = {**json1, **json2}

    with open(os.path.join(output_folder, 'participants.json'), 'w') as outfile:
        json.dump(json_merged, outfile, indent=4)


    # Create a dataset_decription.json -  This is needed on the BIDS loader
    with open(os.path.join(output_folder, 'dataset_description.json'), 'w') as outfile:
        json.dump({"BIDSVersion": "1.0.1", "Name": "SCT_testing"}, outfile, indent=4) # Confirm the version is correct


    all_datasets = [datasetFolder1, datasetFolder2]
    for datasetFolder in all_datasets:
        subjectsFolders = glob.glob(os.path.join(datasetFolder, 'sub-*'))
        derivativesFolder = glob.glob(os.path.join(datasetFolder, 'derivatives'))

        if derivativesFolder!=[]:
            foldersToCopy = subjectsFolders.append(derivativesFolder)
            print("No derivatives are present in this folder")
        else:
            foldersToCopy = subjectsFolders

        for subFolder in foldersToCopy:
            copy_tree(subFolder, os.path.join(output_folder, os.path.basename(os.path.normpath(subFolder))))


if __name__ == "__main__":
    main_run(sys.argv[1])

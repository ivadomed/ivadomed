from distutils.dir_util import copy_tree
import pandas as pd
import json
import glob
import os
import argparse
from ivadomed import utils as imed_utils
import random


# This scripts merges several BIDS datasets, with the optional selection of copying based on metadata
# the new participants.tsv and participants.json are merged versions of the initial files.

# Example call:
# python3 merge_merge_bids_datasets.py --ifolders ~/first_Dataset/ ~/second_Dataset/ --ofolder ~/mergedDataset/
#                                      --metadata institute_id MGH --nsubjects 20 --copy_files False

# Konstantinos Nasiotis 2021

# -----------------------------------------------------------------------------------------------------------------------#


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ifolders", required=True,
                        help="Input BIDS folders.", nargs="*",
                        metavar=imed_utils.Metavar.list)
    parser.add_argument("-o", "--ofolder", required=False, default="./merged_bids_datasets",
                        help="Output folder for merged datasets.",
                        metavar=imed_utils.Metavar.folder)
    parser.add_argument("-m", "--metadata", required=False, default=[], nargs=2,
                        help="Select files within a dataset that match specific metadata.",
                        metavar=imed_utils.Metavar.list)
    parser.add_argument("--nsubjects", required=False, default=[],
                        help="Select the total number of subjects to have on the merged dataset",
                        metavar=imed_utils.Metavar.int)
    parser.add_argument("--copy_files", required=False, default=True,
                        help="Copy the files from the original folders to the output folder.",
                        metavar=imed_utils.Metavar.bool)
    return parser


def run_merging(input_folders, output_folder, metadata=[], nsubjects=[], copy_files=True):

    """
    # This scripts merges several BIDS datasets, with the optional selection of copying based on metadata
    # the new participants.tsv and participants.json are merged versions of the initial files.

    # Usage example::
    # python3 merge_merge_bids_datasets.py --ifolders ~/first_Dataset/ ~/second_Dataset/ --ofolder ~/mergedDataset/
        --metadata institute_id MGH --nsubjects 20 --copy_files False

    Args:
        input_folders (list): list of the Datasets / BIDS Folders to be merged, Flag: ``--ifolders``
        output_folder (str): output folder to of the merged datasets, Flag: ``--ofolder``
        metadata (list) - optional:  2 elements - (1) column label of the participants.csv metadata so only subjects that belong to that category will be used
                                    (2) string to be matched, Flag: ``--metadata``, Example: "--metadata pathology ms"
        nsubjects (int): total number of subjects to have on the merged dataset. They will randomly be selected from the input folders
        copy_files (boolean) - optional: Optional flag to not copy the nifti files (just created merged .tsv, .json files)
    """

    # access CLI options
    print("Input folders: %r" % input_folders)
    print("Output folder: %r" % output_folder)
    if metadata:
        print("Metadata Selection: " + metadata[0] + " - " + metadata[1])
    if nsubjects:
        print("The merged dataset will have maximum " + str(nsubjects) + " subjects")

    # Create output folder if it doesnt exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Create a dataset_decription.json -  This is needed on the BIDS loader
    # Confirm the version is correct
    with open(os.path.join(output_folder, 'dataset_description.json'), 'w') as outfile:
        json.dump({"BIDSVersion": "1.0.1", "Name": "Ivadomed BIDS Merger"}, outfile, indent=4)

    # Checks
    if isinstance(input_folders, str):
        raise TypeError("'ifolders' should be a list")
    elif len(input_folders) < 2:
        raise Exception("Less than 2 datasets selected. No need to merge")

    # Merge multiple .tsv files into the same dataframe
    df_merged = pd.read_table(os.path.join(input_folders[0], 'participants.tsv'))
    # Convert to string to get rid of potential TypeError during merging within the same column
    df_merged = df_merged.astype(str)

    # Get first .json file
    jsonFile = os.path.join(input_folders[0], 'participants.json')
    with open(jsonFile) as json_file:
        json_merged = json.load(json_file)

    for iFolder in range(1, len(input_folders)):
        df_next = pd.read_table(os.path.join(input_folders[iFolder], 'participants.tsv'))
        df_next = df_next.astype(str)
        # Merge the .tsv files (This keeps also non-overlapping fields)
        df_merged = pd.merge(left=df_merged, right=df_next, how='outer')

        jsonFile_next = os.path.join(input_folders[iFolder], 'participants.json')
        with open(jsonFile_next) as json_file:
            jsonNext = json.load(json_file)
        # Merge .json files
        json_merged = {**json_merged, **jsonNext}

    with open(os.path.join(output_folder, 'participants.json'), 'w') as outfile:
        json.dump(json_merged, outfile, indent=4)

    if copy_files:
        subjects_to_copy = []
        derivatives_to_copy = []
        # Start copying folders and files
        for datasetFolder in input_folders:
            subjectsFolders = glob.glob(os.path.join(datasetFolder, 'sub-*'))
            derivativesFolders = glob.glob(os.path.join(datasetFolder, 'derivatives', 'labels', 'sub-*'))

            subjects_to_copy.extend(subjectsFolders)  # Collection from all bids datasets
            derivatives_to_copy.extend(derivativesFolders)

        # In case selection is based on metadata
        if metadata:
            if metadata[0] not in df_merged.keys():
                raise NameError('Invalid selection of Metadata key')
            else:
                if not df_merged[metadata[0]].str.contains(metadata[1]).any():
                    raise NameError('No subjects meet the metadata criteria selected')

            selected_df = df_merged[df_merged[metadata[0]].str.contains(metadata[1])]
            selected_subjects = [i for i in subjects_to_copy if selected_df["participant_id"].str.contains(os.path.basename(i)).any()]
            selected_derivatives = [i for i in derivatives_to_copy if selected_df["participant_id"].str.contains(os.path.basename(i)).any()]
            # Update folders and tsv based on the selection
            subjects_to_copy = selected_subjects
            derivatives_to_copy = selected_derivatives
            df_merged = selected_df

        # If specific number of subjects is selected for merged dataset
        if nsubjects:
            if len(subjects_to_copy) > nsubjects:
                selected_subjects = random.sample(subjects_to_copy, nsubjects)
                selected_derivatives = [os.path.join(os.path.dirname(i), 'derivatives', 'labels', os.path.basename(i))
                                        for i in selected_subjects]
                # Update the folders to be copied
                subjects_to_copy = selected_subjects
                derivatives_to_copy = selected_derivatives
                # Update the participants.tsv based on the selection
                selected_df = []
                for subject in selected_subjects:
                    selected_df.append(df_merged[df_merged["participant_id"] == os.path.basename(subject)])
                df_merged = pd.concat(selected_df)

        for subFolder in subjects_to_copy:
            copy_tree(subFolder, os.path.join(output_folder, os.path.basename(subFolder)))
        for subFolder in derivatives_to_copy:
            copy_tree(subFolder, os.path.join(output_folder, "derivatives", "labels", os.path.basename(subFolder)))

        # Create new participants.tsv from the merged entries
        df_merged.to_csv(os.path.join(output_folder, 'participants.tsv'), sep=',', index=False)


def main(args=None):
    imed_utils.init_ivadomed()
    parser = get_parser()
    args = imed_utils.get_arguments(parser, args)
    run_merging(input_folders=args.ifolders, output_folder=args.ofolder, metadata=args.metadata,
                nsubjects=int(args.nsubjects), copy_files=args.copy_files)


if __name__ == '__main__':
    main()

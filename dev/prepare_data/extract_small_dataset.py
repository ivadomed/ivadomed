#!/usr/bin/env python

import os
import shutil
import random
import argparse
import pandas as pd

EXCLUDED_SUBJECT = ["sub-mniPilot1"]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input BIDS folder.")
    parser.add_argument("-n", "--number", required=False, default=1,
                        help="Number of subjects.")
    parser.add_argument("-c", "--contrasts", required=False,
                        help="Contrast list.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output BIDS Folder.")
    parser.add_argument("-s", "--seg", required=False, default=1,
                        help="1: Include derivatives/labels content. 0: Do not include derivatives/labels content.")
    return parser


def is_good_contrast(fname, good_contrast_list):
    for good_contrast in good_contrast_list:
        if "_" + good_contrast in fname:
            return True
    return False


def remove_some_contrasts(folder, subject_list, good_contrast_list):
    file_list = [os.path.join(folder, s, "anat", f) for s in subject_list
                 for f in os.listdir(os.path.join(folder, s, "anat"))]
    rm_file_list = [f for f in file_list if not is_good_contrast(f, good_contrast_list)]
    for ff in rm_file_list:
        os.remove(ff)


def extract_small_dataset(ifolder, ofolder, n=10, contrast_list=None, include_derivatives=True):
    # Create o folders
    if not os.path.isdir(ofolder):
        os.makedirs(ofolder)
    if include_derivatives:
        oderivatives = os.path.join(ofolder, "derivatives")
        if not os.path.isdir(oderivatives):
            os.makedirs(oderivatives)
        oderivatives = os.path.join(oderivatives, "labels")
        if not os.path.isdir(oderivatives):
            os.makedirs(oderivatives)
        iderivatives = os.path.join(ifolder, "derivatives", "labels")

    # Get subject list
    subject_list = [s for s in os.listdir(ifolder)
                    if s.startswith("sub-") and os.path.isdir(os.path.join(ifolder, s)) and not s in EXCLUDED_SUBJECT]
    # Randomly select subjects
    subject_random_list = random.sample(subject_list, n)

    # Loop across subjects
    for subject in subject_random_list:
        print("\nSubject: {}".format(subject))
        # Copy images
        isubjfolder = os.path.join(ifolder, subject)
        osubjfolder = os.path.join(ofolder, subject)
        assert os.path.isdir(isubjfolder)
        print("\tCopying {} to {}.".format(isubjfolder, osubjfolder))
        shutil.copytree(isubjfolder, osubjfolder)
        # Remove dwi data
        if os.path.isdir(os.path.join(ofolder, subject, "dwi")):
            shutil.rmtree(os.path.join(ofolder, subject, "dwi"))
        # Copy labels
        if include_derivatives:
            isubjderivatives = os.path.join(iderivatives, subject)
            osubjderivatives = os.path.join(oderivatives, subject)
            assert os.path.isdir(isubjderivatives)
            print("\tCopying {} to {}.".format(isubjderivatives, osubjderivatives))
            shutil.copytree(isubjderivatives, osubjderivatives)
            # Remove dwi data
            if os.path.isdir(os.path.join(osubjderivatives, subject, "dwi")):
                shutil.rmtree(os.path.join(osubjderivatives, subject, "dwi"))

    if contrast_list:
        remove_some_contrasts(ofolder, subject_random_list, contrast_list)
        if include_derivatives:
            remove_some_contrasts(os.path.join(ofolder, "derivatives", "labels"), subject_random_list, contrast_list)

    # Copy dataset_description.json
    idatasetjson = os.path.join(ifolder, "dataset_description.json")
    odatasetjson = os.path.join(ofolder, "dataset_description.json")
    shutil.copyfile(idatasetjson, odatasetjson)
    # Copy participants.json
    iparticipantsjson = os.path.join(ifolder, "participants.json")
    oparticipantsjson = os.path.join(ofolder, "participants.json")
    shutil.copyfile(iparticipantsjson, oparticipantsjson)
    # Copy participants.tsv
    iparticipantstsv = os.path.join(ifolder, "participants.tsv")
    oparticipantstsv = os.path.join(ofolder, "participants.tsv")
    df = pd.read_csv(iparticipantstsv, sep='\t')
    # Drop subjects
    df = df[df.participant_id.isin(subject_random_list)]
    df.to_csv(oparticipantstsv, sep='\t', index=False)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # Run script
    extract_small_dataset(args.input, args.output, int(args.number), args.contrasts.split(","), bool(int(args.seg)))
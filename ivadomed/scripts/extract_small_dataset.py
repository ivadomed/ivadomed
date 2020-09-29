#!/usr/bin/env python

import os
import shutil
import argparse
import numpy as np
import pandas as pd

from ivadomed.utils import init_ivadomed

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
    parser.add_argument("-s", "--seed", required=False, default=-1,
                        help="Set np.random.RandomState to ensure reproducibility: the same subjects will be selected "
                             "if the script is run several times on the same dataset. Set to -1 (default) otherwise.")
    parser.add_argument("-d", "--derivatives", required=False, default=1,
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


def extract_small_dataset(input, output, n=10, contrast_list=None, include_derivatives=True, seed=-1):
    """Extract small BIDS dataset from a larger BIDS dataset.

    Example::

         ivadomed_extract_small_dataset -i path/to/BIDS/dataset -o path/of/small/BIDS/dataset -n 10 -c T1w,T2w -d 0 -s 1234

    Args:
        input (str): Input BIDS folder. Flag: ``--input``, ``-i``
        output (str): Output folder. Flag: ``--output``, ``-o``
        n (int): Number of subjects in the output folder. Flag: ``--number``, ``-n``
        contrast_list (list): List of image contrasts to include. If set to None, then all available contrasts are
            included. Flag: ``--contrasts``, ``-c``
        include_derivatives (bool): If True, derivatives/labels/ content is also copied, only the raw images otherwise.
                                    Flag: ``--derivatives``, ``-d``
        seed (int): Set np.random.RandomState to ensure reproducibility: the same subjects will be selected if the
            function is run several times on the same dataset. If set to -1, each function run is independent.
            Flag: ``--seed``, ``-s``.
    """
    # Create o folders
    if not os.path.isdir(output):
        os.makedirs(output)
    if include_derivatives:
        oderivatives = os.path.join(output, "derivatives")
        if not os.path.isdir(oderivatives):
            os.makedirs(oderivatives)
        oderivatives = os.path.join(oderivatives, "labels")
        if not os.path.isdir(oderivatives):
            os.makedirs(oderivatives)
        iderivatives = os.path.join(input, "derivatives", "labels")

    # Get subject list
    subject_list = [s for s in os.listdir(input)
                    if s.startswith("sub-") and os.path.isdir(os.path.join(input, s)) and not s in EXCLUDED_SUBJECT]
    # Randomly select subjects
    if seed != -1:
        # Reproducibility
        r = np.random.RandomState(seed)
        subject_random_list = list(r.choice(subject_list, n))
    else:
        subject_random_list = list(np.random.choice(subject_list, n))

    # Loop across subjects
    for subject in subject_random_list:
        print("\nSubject: {}".format(subject))
        # Copy images
        isubjfolder = os.path.join(input, subject)
        osubjfolder = os.path.join(output, subject)
        assert os.path.isdir(isubjfolder)
        print("\tCopying {} to {}.".format(isubjfolder, osubjfolder))
        shutil.copytree(isubjfolder, osubjfolder)
        # Remove dwi data
        if os.path.isdir(os.path.join(output, subject, "dwi")):
            shutil.rmtree(os.path.join(output, subject, "dwi"))
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
        remove_some_contrasts(output, subject_random_list, contrast_list)
        if include_derivatives:
            remove_some_contrasts(os.path.join(output, "derivatives", "labels"), subject_random_list, contrast_list)

    # Copy dataset_description.json
    idatasetjson = os.path.join(input, "dataset_description.json")
    odatasetjson = os.path.join(output, "dataset_description.json")
    shutil.copyfile(idatasetjson, odatasetjson)
    # Copy participants.json if it exist
    if os.path.isfile(os.path.join(input, "participants.json")):
        iparticipantsjson = os.path.join(input, "participants.json")
        oparticipantsjson = os.path.join(output, "participants.json")
        shutil.copyfile(iparticipantsjson, oparticipantsjson)
    # Copy participants.tsv
    iparticipantstsv = os.path.join(input, "participants.tsv")
    oparticipantstsv = os.path.join(output, "participants.tsv")
    df = pd.read_csv(iparticipantstsv, sep='\t')
    # Drop subjects
    df = df[df.participant_id.isin(subject_random_list)]
    df.to_csv(oparticipantstsv, sep='\t', index=False)


def main():
    init_ivadomed()

    parser = get_parser()
    args = parser.parse_args()
    # Run script
    extract_small_dataset(args.input, args.output, int(args.number), args.contrasts.split(","),
                          bool(int(args.derivatives)), int(args.seed))


if __name__ == '__main__':
    main()

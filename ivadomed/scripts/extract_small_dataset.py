#!/usr/bin/env python

import shutil
import argparse
import numpy as np
import pandas as pd
from ivadomed import utils as imed_utils
from pathlib import Path

EXCLUDED_SUBJECT = ["sub-mniPilot1"]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input BIDS folder.", metavar=imed_utils.Metavar.file)
    parser.add_argument("-n", "--number", required=False, default=1,
                        help="Number of subjects.", metavar=imed_utils.Metavar.int)
    parser.add_argument("-c", "--contrasts", required=False,
                        help="Contrast list.", metavar=imed_utils.Metavar.list)
    parser.add_argument("-o", "--output", required=True,
                        help="Output BIDS Folder.", metavar=imed_utils.Metavar.file)
    parser.add_argument("-s", "--seed", required=False, default=-1,
                        help="""Set np.random.RandomState to ensure reproducibility: the same
                                subjects will be selected if the script is run several times on the
                                same dataset. Set to -1 (default) otherwise.""",
                        metavar=imed_utils.Metavar.int)
    parser.add_argument("-d", "--derivatives",
                        dest="derivatives",
                        default=1,
                        help="""If true, include derivatives/labels content.
                                1 = true, 0 = false""",
                        metavar=imed_utils.Metavar.int)
    return parser


def is_good_contrast(fname, good_contrast_list):
    for good_contrast in good_contrast_list:
        if "_" + good_contrast in fname:
            return True
    return False


def remove_some_contrasts(folder, subject_list, good_contrast_list):
    file_list = [Path(folder).joinpath(s, "anat", f) for s in subject_list
                 for f in Path(folder).joinpath(s, "anat").iterdir()]
    rm_file_list = [f for f in file_list if not is_good_contrast(str(f), good_contrast_list)]
    for ff in rm_file_list:
        ff.unlink()


def extract_small_dataset(input, output, n=10, contrast_list=None, include_derivatives=True,
                          seed=-1):
    """Extract small BIDS dataset from a larger BIDS dataset.

    Example::

         ivadomed_extract_small_dataset -i path/to/BIDS/dataset -o path/of/small/BIDS/dataset \
            -n 10 -c T1w,T2w -d 0 -s 1234

    Args:
        input (str): Input BIDS folder. Flag: ``--input``, ``-i``
        output (str): Output folder. Flag: ``--output``, ``-o``
        n (int): Number of subjects in the output folder. Flag: ``--number``, ``-n``
        contrast_list (list): List of image contrasts to include. If set to None, then all
            available contrasts are included. Flag: ``--contrasts``, ``-c``
        include_derivatives (bool): If True, derivatives/labels/ content is also copied,
            only the raw images otherwise. Flag: ``--derivatives``, ``-d``
        seed (int): Set np.random.RandomState to ensure reproducibility: the same subjects will be
            selected if the function is run several times on the same dataset. If set to -1,
            each function run is independent. Flag: ``--seed``, ``-s``.
    """
    # Create output folders
    if not Path(output).is_dir():
        Path(output).mkdir(parents=True)
    if include_derivatives:
        oderivatives = Path(output).joinpath("derivatives")
        if not oderivatives.is_dir():
            oderivatives.mkdir(parents=True)
        oderivatives = oderivatives.joinpath("labels")
        if not oderivatives.is_dir():
            oderivatives.mkdir(parents=True)
        iderivatives = Path(input).joinpath("derivatives", "labels")

    # Get subject list
    subject_list = [s.name for s in Path(input).iterdir()
                    if s.name.startswith("sub-") and s.is_dir()
                    and s.name not in EXCLUDED_SUBJECT]

    # Randomly select subjects
    if seed != -1:
        # Reproducibility
        r = np.random.RandomState(seed)
        subject_random_list = list(r.choice(subject_list, n))
    else:
        subject_random_list = list(np.random.choice(subject_list, n, replace=False))

    # Loop across subjects
    for subject in subject_random_list:
        print("\nSubject: {}".format(subject))
        # Copy images
        isubjfolder = Path(input).joinpath(subject)
        osubjfolder = Path(output).joinpath(subject)
        assert isubjfolder.is_dir()
        print("\tCopying {} to {}.".format(isubjfolder, osubjfolder))
        shutil.copytree(str(isubjfolder), str(osubjfolder))
        # Remove dwi data
        if Path(output).joinpath(subject, "dwi").is_dir():
            shutil.rmtree(str(Path(output).joinpath(subject, "dwi")))
        # Copy labels
        if include_derivatives:
            isubjderivatives = iderivatives.joinpath(subject)
            osubjderivatives = oderivatives.joinpath(subject)
            assert isubjderivatives.is_dir()
            print("\tCopying {} to {}.".format(isubjderivatives, osubjderivatives))
            shutil.copytree(str(isubjderivatives), str(osubjderivatives))
            # Remove dwi data
            if osubjderivatives.joinpath(subject, "dwi").is_dir():
                shutil.rmtree(str(osubjderivatives.joinpath(subject, "dwi")))

    if contrast_list:
        remove_some_contrasts(output, subject_random_list, contrast_list)
        if include_derivatives:
            remove_some_contrasts(str(Path(output).joinpath("derivatives", "labels")),
                                  subject_random_list, contrast_list)

    # Copy dataset_description.json
    idatasetjson = Path(input).joinpath("dataset_description.json")
    odatasetjson = Path(output).joinpath("dataset_description.json")
    shutil.copyfile(str(idatasetjson), str(odatasetjson))
    # Copy participants.json if it exist
    if Path(input).joinpath("participants.json").is_file():
        iparticipantsjson = Path(input).joinpath("participants.json")
        oparticipantsjson = Path(output).joinpath("participants.json")
        shutil.copyfile(str(iparticipantsjson), str(oparticipantsjson))
    # Copy participants.tsv
    iparticipantstsv = Path(input).joinpath("participants.tsv")
    oparticipantstsv = Path(output).joinpath("participants.tsv")
    df = pd.read_csv(str(iparticipantstsv), sep='\t')
    # Drop subjects
    df = df[df.participant_id.isin(subject_random_list)]
    df.to_csv(str(oparticipantstsv), sep='\t', index=False)


def main(args=None):
    imed_utils.init_ivadomed()
    parser = get_parser()
    args = imed_utils.get_arguments(parser, args)
    if args.contrasts is not None:
        contrast_list = args.contrasts.split(",")
    else:
        contrast_list = None

    extract_small_dataset(args.input, args.output, int(args.number), contrast_list,
                          bool(int(args.derivatives)), int(args.seed))


if __name__ == '__main__':
    main()

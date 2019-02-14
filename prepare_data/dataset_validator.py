#!/usr/bin/env python
#
# Check if dataset is named correctly for BIDS.
#
# Usage:
#   python dataset_validator.py -d BIDS_FOLDER
#
# Authors: Alexandru Foias, Julien Cohen-Adad

import os, argparse

def get_parameters():
    parser = argparse.ArgumentParser(description='Check if dataset is named correctly for BIDS ')
    parser.add_argument('-d', '--path-data',
                        help='Path to input BIDS dataset directory.',
                        required=True)
    args = parser.parse_args()
    return args


def check_bids_dataset(path_data):
    """
    Check if dataset is named correctly for BIDS.
    :param path_data: Path to input BIDS dataset directory
    :return:
    """

    # Dictionary of BIDS naming. First element: file name suffix, Second element: destination folder.
    # Note: this dictionary is based on the spine_generic protocol, but could be extended to other usage:
    contrast_dict = {
        'GRE-MT0': ('acq-MToff_MTS', 'anat'),
        'GRE-MT1': ('acq-MTon_MTS', 'anat'),
        'GRE-T1w': ('acq-T1w_MTS', 'anat'),
        'GRE-ME': ('T2star', 'anat'),
        'T1w': ('T1w', 'anat'),
        'T2w': ('T2w', 'anat'),
        'DWI': ('dwi', 'dwi'),
    }

    list_items_dataset =  os.listdir(path_data)
    #loop across subjects within a BIDS dataset
    for sub_data in list_items_dataset:
        path_sub_data = os.path.join(path_data,sub_data)
        if os.path.isdir (path_sub_data):
            #looping across contrast withn a subject
            for contrast in list(contrast_dict.keys()):
                path_contrast = os.path.join(path_sub_data,contrast_dict[contrast][1],sub_data + '_' +contrast_dict[contrast][0])
                path_contrast_nii = os.path.join(path_contrast + '.nii.gz')
                path_contrast_json = os.path.join(path_contrast + '.json')

                if os.path.isfile(path_contrast_nii) == False:
                    print 'Warning - missing : ' + path_contrast_nii
                if os.path.isfile(path_contrast_json) == False:
                    print 'Warning - missing : ' + path_contrast_json

if __name__ == "__main__":
    args = get_parameters()
    check_bids_dataset(args.path_data)
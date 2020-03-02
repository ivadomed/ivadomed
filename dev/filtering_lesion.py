##############################################################
#
# TODO
#
##############################################################

import os
import json
import argparse
import numpy as np
import nibabel as nib

from ivadomed.main import cmd_test

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Config file path.")
    parser.add_argument("-ofolder", help="Output folder.")

    return parser

def run_main(args):

    with open(args.c, "r") as fhandle:
        context = json.load(fhandle)

    ofolder = args.ofolder
    if not os.path.isdir(ofolder):
        os.makedirs(ofolder)

    pred_folder = os.path.join(context['log_directory'], 'pred_masks')
    if not os.path.isdir(pred_folder):
        cmd_test(context)

    subj_acq_lst = list(set([f.split('_pred')[0] for f in os.listdir(pred_folder)
                    if f.endswith('.nii.gz') and '_pred' in f]))

    metric_suffix_lst = ['_unc-vox', '_unc-cv', '_unc-avgUnc']
    for metric in metric_lst:
        print(metric)
        for subj_acq in subj_acq_lst:
            fname = os.path.join(pred_folder, subj_acq+metric+'.nii.gz')
            im = nib.load(fname)
            data = im.get_data()
            vals = list(data[np.non_zero(data)])
            print(np.min(vals), np.mean(vals), np.median(vals), np.max(vals))

if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    run_main(arguments)

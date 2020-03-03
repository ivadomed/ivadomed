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
from ivadomed.utils import threshold_predictions

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
                    if f.endswith('.nii.gz') and '_pred' in f]))[:5]

    metric_suffix_lst = ['_unc-vox', '_unc-cv', '_unc-avgUnc']
    thr_unc_lst = [0.01, 0.1, 0.5]
    thr_vox_lst = [t/10. for t in range(0,10,1)]
    results_dct = {'metric': {}}
    for metric in metric_suffix_lst:
        print(metric)

        res_init_lst = [[[]] * len(thr_vox_lst)] * len(thr_unc_lst)
        results_dct['metric'][metric] = {'tpr_vox': res_init_lst,
                                            'fdr_vox': res_init_lst,
                                            'tpr_obj': res_init_lst,
                                            'fdr_obj': res_init_lst}

        for subj_acq in subj_acq_lst:
            fname_unc = os.path.join(pred_folder, subj_acq+metric+'.nii.gz')
            im = nib.load(fname)
            data_unc = im.get_data()
            del im

            data_pred_lst = [nib.load(os.path.join(pred_folder, f)).get_data()
                                for f in os.listdir(pred_folder) if subj_acq+'_pred_' in f]

            # data_gt = 

            for i_unc, thr_unc in enumerate(thr_unc_lst):
                data_unc_thr = (data_unc > thr_unc).astype(np.int)

                data_pred_thrUnc_lst = [d * data_unc_thr for d in data_pred_lst]

                data_prob = np.mean(np.array(data_pred_thrUnc_lst), axis=0)

                for i_vox, thr_vox in enumerate(thr_vox_lst):
                    data_hard = threshold_predictions(data_prob, thr=thr_vox).astype(np.uint8)

                    # tpr_vox =
                    # fdr_vox =
                    # tpr_obj =
                    # fdr_obj =

                    results_dct['metric'][metric]['tpr_vox'][i_unc][i_vox].append(tpr_vox)
                    results_dct['metric'][metric]['fdr_vox'][i_unc][i_vox].append(fdr_vox)
                    results_dct['metric'][metric]['tpr_obj'][i_unc][i_vox].append(tpr_obj)
                    results_dct['metric'][metric]['fdr_obj'][i_unc][i_vox].append(fdr_obj)



if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    run_main(arguments)

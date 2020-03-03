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
from copy import deepcopy
from sklearn.metrics import auc
from scipy.ndimage import label, generate_binary_structure

from ivadomed.main import cmd_test
from ivadomed.utils import threshold_predictions
from medicaltorch import metrics as mt_metrics

BIN_STRUCT = generate_binary_structure(3,2)
MIN_OBJ_SIZE = 3

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Config file path.")
    parser.add_argument("-ofolder", help="Output folder.")

    return parser

# TODO: remove once moved out of Evaluation3Dmetrics
def remove_small_obj(data, size_min, bin_struct):
    data_label, n = label(data,
                          structure=bin_struct)

    for idx in range(1, n + 1):
        data_idx = (data_label == idx).astype(np.int)
        n_nonzero = np.count_nonzero(data_idx)

        if n_nonzero < size_min:
            data[data_label == idx] = 0

    return data


# def auc_homemade(fpr, tpr, trapezoid=False):
#     # source: https://stackoverflow.com/a/39687168

#     inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1: ])) if s != e] + [len(fpr) - 1]
#     fpr, tpr = fpr[inds], tpr[inds]
#     area = 0
#     ft = zip(fpr, tpr)
#     for p0, p1 in list(zip(ft[: -1], ft[1: ])):
#         area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
#     return area

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

    gt_folder = os.path.join(context['bids_path'], 'derivatives', 'labels')

    metric_suffix_lst = ['_unc-vox', '_unc-cv', '_unc-avgUnc']
    thr_unc_lst = [0.01, 0.1, 0.5]
    thr_vox_lst = [t/10. for t in range(0,10,1)]
    results_dct = {}
    for metric in metric_suffix_lst:

        tmp_lst = [[] for _ in range(len(thr_vox_lst))]
        res_init_lst = [deepcopy(tmp_lst) for _ in range(len(thr_unc_lst))]

        results_dct[metric] = {'tpr_vox': deepcopy(res_init_lst),
                                'fdr_vox': deepcopy(res_init_lst),
                                # TODO: modify ltpr and lfdr so that callable here
                                # 'tpr_obj': deepcopy(res_init_lst),
                                # 'fdr_obj': deepcopy(res_init_lst)
                                }

        for subj_acq in subj_acq_lst:
            fname_unc = os.path.join(pred_folder, subj_acq+metric+'.nii.gz')
            im = nib.load(fname_unc)
            data_unc = im.get_data()
            del im

            data_pred_lst = [nib.load(os.path.join(pred_folder, f)).get_data()
                                for f in os.listdir(pred_folder) if subj_acq+'_pred_' in f]

            fname_gt = os.path.join(gt_folder, subj_acq.split('_')[0], 'anat', subj_acq+context["target_suffix"]+'.nii.gz')
            if os.path.isfile(fname_gt):
                data_gt = nib.load(fname_gt).get_data()
                data_gt = remove_small_obj(data_gt, MIN_OBJ_SIZE, BIN_STRUCT)

                for i_unc, thr_unc in enumerate(thr_unc_lst):
                    data_unc_thr = (deepcopy(data_unc) > thr_unc).astype(np.int)

                    data_pred_thrUnc_lst = [d * deepcopy(data_unc_thr) for d in data_pred_lst]

                    data_prob = np.mean(np.array(data_pred_thrUnc_lst), axis=0)

                    for i_vox, thr_vox in enumerate(thr_vox_lst):
                        data_hard = threshold_predictions(deepcopy(data_prob), thr=thr_vox).astype(np.uint8)

                        data_hard = remove_small_obj(data_hard, MIN_OBJ_SIZE, BIN_STRUCT)

                        tpr_vox = mt_metrics.recall_score(data_hard, data_gt, err_value=np.nan)
                        fdr_vox = 100. - mt_metrics.precision_score(data_hard, data_gt, err_value=np.nan)
                        #tpr_obj =
                        #fdr_obj =

                        results_dct[metric]['tpr_vox'][i_unc][i_vox].append(tpr_vox / 100.)
                        results_dct[metric]['fdr_vox'][i_unc][i_vox].append(fdr_vox / 100.)
                        #results_dct[metric]['tpr_obj'][i_unc][i_vox].append(tpr_obj)
                        #results_dct[metric]['fdr_obj'][i_unc][i_vox].append(fdr_obj)


    for metric in metric_suffix_lst:
        print('Metric: {}'.format(metric))
        for i_unc, thr_unc in enumerate(thr_unc_lst):
            print('Unc Thr: {}'.format(thr_unc))
            tpr_vals = np.array([np.nanmean(results_dct[metric]['tpr_vox'][i_unc][i_vox]) for i_vox in range(len(thr_vox_lst))])
            fdr_vals = np.array([np.nanmean(results_dct[metric]['fdr_vox'][i_unc][i_vox]) for i_vox in range(len(thr_vox_lst))])

            auc_ = auc(fdr_vals, tpr_vals)          
            # auc_ = auc_homemade(fdr_vals, tpr_vals)
            # auc__ = auc_homemade(fdr_vals, tpr_vals, True)
            
            optimal_idx = np.argmax(tpr_vals - fdr_vals)
            optimal_threshold = thr_vox_lst[optimal_idx]
            print('AUC: {}, Optimal Pred Thr: {}'.format(auc_, optimal_threshold))

if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    run_main(arguments)

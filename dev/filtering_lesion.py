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
#from sklearn.metrics import auc
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure

from ivadomed.main import cmd_test
from ivadomed.utils import threshold_predictions, Evaluation3DMetrics
from medicaltorch import metrics as mt_metrics

BIN_STRUCT = generate_binary_structure(3,2)
MIN_OBJ_SIZE = 3

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Config file path.")
    parser.add_argument("-ofolder", help="Output folder.")

    return parser

def auc_homemade(fpr, tpr, trapezoid=False):
    # source: https://stackoverflow.com/a/39687168
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1: ])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = list(zip(fpr, tpr))
    for p0, p1 in zip(ft[: -1], ft[1: ]):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area

def run_experiment(level, unc_name, thr_unc_lst, thr_pred_lst, gt_folder, pred_folder, im_lst):

    # init results
    tmp_lst = [[] for _ in range(len(thr_pred_lst))]
    res_init_lst = [deepcopy(tmp_lst) for _ in range(len(thr_unc_lst))]
    res_dct = {'tpr': deepcopy(res_init_lst),
                  'fdr': deepcopy(res_init_lst),
                  'retained_elt': [[] for _ in range(len(thr_unc_lst))]
    }

    for fname_pref in im_lst:
        fname_unc = os.path.join(pred_folder, fname_pref+unc_name+'.nii.gz')
        im = nib.load(fname_unc)
        data_unc = im.get_data()
        print(np.percentile(data_unc, 25), np.median(data_unc), np.percentile(data_unc, 75), np.max(data_unc))
        del im

"""
            data_pred_lst = np.array([nib.load(os.path.join(pred_folder, f)).get_data()
                                for f in os.listdir(pred_folder) if subj_acq+'_pred_' in f])

            fname_gt = os.path.join(gt_folder, subj_acq.split('_')[0], 'anat', subj_acq+context["target_suffix"]+'.nii.gz')
            if os.path.isfile(fname_gt):
                nib_gt = nib.load(fname_gt)
                data_gt = nib_gt.get_data()

                for i_unc, thr_unc in enumerate(thr_unc_lst):
                    data_prob = np.mean(data_pred_lst, axis=0)

                    data_prob_thrUnc = deepcopy(data_prob)
                    data_prob_thrUnc[data_unc > thr_unc] = 0

                    cmpt_vox_beforeThr = np.count_nonzero(data_prob)
                    cmpt_vox_afterThr = np.count_nonzero(data_prob_thrUnc)
                    percent_rm_vox = (cmpt_vox_beforeThr - cmpt_vox_afterThr) * 100. / cmpt_vox_beforeThr
                    percent_retained_vox = 100. - percent_rm_vox

                    _, n_beforeThr = label((data_prob > 0).astype(np.int), structure=BIN_STRUCT)
                    _, n_afterThr = label((data_prob_thrUnc > 0).astype(np.int), structure=BIN_STRUCT)
                    percent_retained_obj = 100. - ((n_beforeThr - n_afterThr) * 100. / n_beforeThr)

                    results_dct[metric]['retained_vox'][i_unc].append(percent_retained_vox)
                    results_dct[metric]['retained_obj'][i_unc].append(percent_retained_obj)
        print(results_dct[metric]['retained_obj'], results_dct[metric]['retained_vox'])
"""

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
                    if f.endswith('.nii.gz') and '_pred' in f]))[:10]

    gt_folder = os.path.join(context['bids_path'], 'derivatives', 'labels')

    # experiments
    exp_dct = {
                'exp1': {'level': 'vox',
                            'uncertainty_measure': '_unc-vox',
                            'uncertainty_thr': [1e-5, 1e-3, 1e-1, 0.5],
                            'prediction_thr': [1e-6]+[t/10. for t in range(1,10,1)]},
                'exp2': {'level': 'obj',
                            'uncertainty_measure': '_unc-cv',
                            'uncertainty_thr': [1e-5, 1e-3, 1e-1, 0.5],
                            'prediction_thr': [1e-6]+[t/10. for t in range(1,10,1)]},
                'exp3': {'level': 'obj',
                            'uncertainty_measure': '_unc-avgUnc',
                            'uncertainty_thr': [1e-5, 1e-3, 1e-1, 0.5],
                            'prediction_thr': [1e-6]+[t/10. for t in range(1,10,1)]}
    }

    for exp in exp_dct.keys():
        config_dct = exp_dct[exp]
        res = run_experiement(level=config_dct['level'],
                                unc_name=config_dct['measure'],
                                thr_unc_lst=config_dct['uncertainty_thr'],
                                thr_pred_lst=config_dct['prediction_thr'],
                                gt_folder=gt_folder,
                                pred_folder=pred_folder,
                                im_lst=subj_acq_lst)

"""
        for subj_acq in subj_acq_lst:
            fname_unc = os.path.join(pred_folder, subj_acq+metric+'.nii.gz')
            im = nib.load(fname_unc)
            data_unc = im.get_data()
#            print(np.percentile(data_unc, 25), np.median(data_unc), np.percentile(data_unc, 75), np.max(data_unc))
            del im

            data_pred_lst = np.array([nib.load(os.path.join(pred_folder, f)).get_data()
                                for f in os.listdir(pred_folder) if subj_acq+'_pred_' in f])

            fname_gt = os.path.join(gt_folder, subj_acq.split('_')[0], 'anat', subj_acq+context["target_suffix"]+'.nii.gz')
            if os.path.isfile(fname_gt):
                nib_gt = nib.load(fname_gt)
                data_gt = nib_gt.get_data()

                for i_unc, thr_unc in enumerate(thr_unc_lst):
                    data_prob = np.mean(data_pred_lst, axis=0)

                    data_prob_thrUnc = deepcopy(data_prob)
                    data_prob_thrUnc[data_unc > thr_unc] = 0

                    cmpt_vox_beforeThr = np.count_nonzero(data_prob)
                    cmpt_vox_afterThr = np.count_nonzero(data_prob_thrUnc)
                    percent_rm_vox = (cmpt_vox_beforeThr - cmpt_vox_afterThr) * 100. / cmpt_vox_beforeThr
                    percent_retained_vox = 100. - percent_rm_vox

                    _, n_beforeThr = label((data_prob > 0).astype(np.int), structure=BIN_STRUCT)
                    _, n_afterThr = label((data_prob_thrUnc > 0).astype(np.int), structure=BIN_STRUCT)
                    percent_retained_obj = 100. - ((n_beforeThr - n_afterThr) * 100. / n_beforeThr)

                    results_dct[metric]['retained_vox'][i_unc].append(percent_retained_vox)
                    results_dct[metric]['retained_obj'][i_unc].append(percent_retained_obj)
        print(results_dct[metric]['retained_obj'], results_dct[metric]['retained_vox'])
                    for i_vox, thr_vox in enumerate(thr_vox_lst):
                        data_hard = threshold_predictions(deepcopy(data_prob_thrUnc), thr=thr_vox).astype(np.uint8)

                        eval = Evaluation3DMetrics(data_pred=data_hard,
                                                    data_gt=data_gt,
                                                    dim_lst=nib_gt.header['pixdim'][1:4],
                                                    params=context['eval_params'])

                        tpr_vox = mt_metrics.recall_score(eval.data_pred, eval.data_gt, err_value=np.nan)
                        fdr_vox = 100. - mt_metrics.precision_score(eval.data_pred, eval.data_gt, err_value=np.nan)
                        tpr_obj, _ = eval.get_ltpr()
                        fdr_obj = eval.get_lfdr()

#                        print(thr_vox, tpr_vox, fdr_vox, tpr_obj, fdr_obj)

                        results_dct[metric]['tpr_vox'][i_unc][i_vox].append(tpr_vox / 100.)
                        results_dct[metric]['fdr_vox'][i_unc][i_vox].append(fdr_vox / 100.)
                        results_dct[metric]['tpr_obj'][i_unc][i_vox].append(tpr_obj / 100.)
                        results_dct[metric]['fdr_obj'][i_unc][i_vox].append(fdr_obj / 100.)

    for metric in metric_suffix_lst:
        print('Metric: {}'.format(metric))

        plt.figure(figsize=(10,10))
        for i_unc, thr_unc in enumerate(thr_unc_lst):
            print('Unc Thr: {}'.format(thr_unc))

            mean_retained_vox = np.mean(results_dct[metric]['retained_vox'][i_unc])
            mean_retained_obj = np.mean(results_dct[metric]['retained_obj'][i_unc])
            print('Mean percentage of retained voxels: {}%, lesions: {}%.'.format(thr_unc, mean_retained_vox, mean_retained_obj))

            tpr_vals = np.array([np.nanmean(results_dct[metric]['tpr_vox'][i_unc][i_vox]) for i_vox in range(len(thr_vox_lst))])
            fdr_vals = np.array([np.nanmean(results_dct[metric]['fdr_vox'][i_unc][i_vox]) for i_vox in range(len(thr_vox_lst))])

            # auc_ = auc(fdr_vals, tpr_vals)
            # auc_ = auc_homemade(fdr_vals, tpr_vals)
            auc_ = auc_homemade(fdr_vals, tpr_vals, True)

            optimal_idx = np.argmax(tpr_vals - fdr_vals)
            optimal_threshold = thr_vox_lst[optimal_idx]
            print('AUC: {}, Optimal Pred Thr: {}'.format(auc_, optimal_threshold))

            plt.plot(fdr_vals, tpr_vals, label='Unc thr={0:0.2f} (area = {1:0.2f})'.format(thr_unc, auc_))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Detection Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - '+metric)
        plt.legend(loc="lower right")
        fname_out = os.path.join(ofolder, metric+'.png')
        plt.savefig(fname_out, bbox_inches='tight', pad_inches=0)
        plt.close()
"""
if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    run_main(arguments)

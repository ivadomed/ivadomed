import os
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm

from ivadomed import utils as imed_utils


def evaluate(bids_path, log_directory, target_suffix, eval_params):
    """Evaluate predictions from inference step.
    
    Args:
          bids_path (string): Folder where data is stored
          log_directory (string): Output folder where predictions were saved
          target_suffix (list): list of suffixes
          eval_params (dict):
    Returns:
          pd.DataFrame: results for each prediction
    """
    # PREDICTION FOLDER
    path_pred = os.path.join(log_directory, 'pred_masks')
    # If the prediction folder does not exist, run Inference first
    """
    if not os.path.isdir(path_pred):
        print('\nRun Inference\n')
        metrics_dict = cmd_test(context)
    """
    print('\nRun Evaluation on {}\n'.format(path_pred))

    # OUTPUT RESULT FOLDER
    path_results = os.path.join(log_directory, 'results_eval')
    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    # INIT DATA FRAME
    df_results = pd.DataFrame()

    # LIST PREDS
    subj_acq_lst = [f.split('_pred')[0] for f in os.listdir(path_pred) if f.endswith('_pred.nii.gz')]

    # LOOP ACROSS PREDS
    for subj_acq in tqdm(subj_acq_lst, desc="Evaluation"):
        # Fnames of pred and ground-truth
        subj, acq = subj_acq.split('_')[0], '_'.join(subj_acq.split('_')[1:])
        fname_pred = os.path.join(path_pred, subj_acq + '_pred.nii.gz')
        fname_gt = [os.path.join(bids_path, 'derivatives', 'labels', subj, 'anat', subj_acq + suffix + '.nii.gz')
                    for suffix in target_suffix]

        # 3D evaluation
        nib_pred = nib.load(fname_pred)
        data_pred = nib_pred.get_fdata()
        h, w, d = data_pred.shape[:3]
        n_classes = len(fname_gt)
        data_gt = np.zeros((h, w, d, n_classes))
        for idx, file in enumerate(fname_gt):
            if os.path.exists(file):
                data_gt[..., idx] = nib.load(file).get_fdata()
            else:
                data_gt[..., idx] = np.zeros((h, w, d), dtype='u1')
        eval = imed_utils.Evaluation3DMetrics(data_pred=data_pred,
                                              data_gt=data_gt,
                                              dim_lst=nib_pred.header['pixdim'][1:4],
                                              params=eval_params)
        results_pred, data_painted = eval.run_eval()

        # SAVE PAINTED DATA, TP FP FN
        fname_paint = fname_pred.split('.nii.gz')[0] + '_painted.nii.gz'
        nib_painted = nib.Nifti1Image(data_painted, nib_pred.affine)
        nib.save(nib_painted, fname_paint)

        # SAVE RESULTS FOR THIS PRED
        results_pred['image_id'] = subj_acq
        df_results = df_results.append(results_pred, ignore_index=True)

    df_results = df_results.set_index('image_id')
    df_results.to_csv(os.path.join(path_results, 'evaluation_3Dmetrics.csv'))

    print(df_results.head(5))
    return df_results
    #return metrics_dict, df_results

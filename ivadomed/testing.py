import os
import copy
import nibabel as nib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from ivadomed import metrics as imed_metrics
from ivadomed import utils as imed_utils
from ivadomed import visualize as imed_visualize
from ivadomed import inference as imed_inference
from ivadomed import uncertainty as imed_uncertainty
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.object_detection import utils as imed_obj_detect
from ivadomed.training import get_metadata
from ivadomed.postprocessing import threshold_predictions

cudnn.benchmark = True


def test(model_params, dataset_test, testing_params, log_directory, device, cuda_available=True,
         metric_fns=None, postprocessing=None):
    """Main command to test the network.

    Args:
        model_params (dict): Model's parameters.
        dataset_test (imed_loader): Testing dataset.
        testing_params (dict): Testing parameters.
        log_directory (str): Folder where predictions are saved.
        device (torch.device): Indicates the CPU or GPU ID.
        cuda_available (bool): If True, CUDA is available.
        metric_fns (list): List of metrics, see :mod:`ivadomed.metrics`.
        postprocessing (dict): Contains postprocessing steps.

    Returns:
        dict: result metrics.
    """
    # DATA LOADER
    test_loader = DataLoader(dataset_test, batch_size=testing_params["batch_size"],
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=0)

    # LOAD TRAIN MODEL
    fname_model = os.path.join(log_directory, "best_model.pt")
    print('\nLoading model: {}'.format(fname_model))
    model = torch.load(fname_model, map_location=device)
    if cuda_available:
        model.cuda()
    model.eval()

    # CREATE OUTPUT FOLDER
    path_3Dpred = os.path.join(log_directory, 'pred_masks')
    if not os.path.isdir(path_3Dpred):
        os.makedirs(path_3Dpred)

    # METRIC MANAGER
    metric_mgr = imed_metrics.MetricManager(metric_fns)

    # UNCERTAINTY SETTINGS
    if (testing_params['uncertainty']['epistemic'] or testing_params['uncertainty']['aleatoric']) and \
            testing_params['uncertainty']['n_it'] > 0:
        n_monteCarlo = testing_params['uncertainty']['n_it'] + 1
        testing_params['uncertainty']['applied'] = True
        print('\nComputing model uncertainty over {} iterations.'.format(n_monteCarlo - 1))
    else:
        testing_params['uncertainty']['applied'] = False
        n_monteCarlo = 1

    for i_monteCarlo in range(n_monteCarlo):
        preds_npy, gt_npy = run_inference(test_loader, model, model_params, testing_params, path_3Dpred,
                                          cuda_available, i_monteCarlo, postprocessing)
        metric_mgr(preds_npy, gt_npy)
        # If uncertainty computation, don't apply it on last iteration for prediction
        if testing_params['uncertainty']['applied'] and (n_monteCarlo - 2 == i_monteCarlo):
            testing_params['uncertainty']['applied'] = False
            # COMPUTE UNCERTAINTY MAPS
            imed_uncertainty.run_uncertainty(ifolder=path_3Dpred)

    metrics_dict = metric_mgr.get_results()
    metric_mgr.reset()
    print(metrics_dict)
    return metrics_dict


def run_inference(test_loader, model, model_params, testing_params, ofolder, cuda_available,
                  i_monte_carlo=None, postprocessing=None):
    """Run inference on the test data and save results as nibabel files.

    Args:
        test_loader (torch DataLoader):
        model (nn.Module):
        model_params (dict):
        testing_params (dict):
        ofolder (str): Folder where predictions are saved.
        cuda_available (bool): If True, CUDA is available.
        i_monte_carlo (int): i_th Monte Carlo iteration.
        postprocessing (dict): Indicates postprocessing steps.

    Returns:
        ndarray, ndarray: Prediction, Ground-truth of shape n_sample, n_label, h, w, d.
    """
    # INIT STORAGE VARIABLES
    preds_npy_list, gt_npy_list, filenames = [], [], []
    pred_tmp_lst, z_tmp_lst, fname_tmp = [], [], ''
    volume = None
    weight_matrix = None

    for i, batch in enumerate(tqdm(test_loader, desc="Inference - Iteration " + str(i_monte_carlo))):
        with torch.no_grad():
            # GET SAMPLES
            # input_samples: list of batch_size tensors, whose size is n_channels X height X width X depth
            # gt_samples: idem with n_labels
            # batch['*_metadata']: list of batch_size lists, whose size is n_channels or n_labels
            if model_params["name"] == "HeMISUnet":
                input_samples = imed_utils.cuda(imed_utils.unstack_tensors(batch["input"]), cuda_available)
            else:
                input_samples = imed_utils.cuda(batch["input"], cuda_available)
            gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)

            # EPISTEMIC UNCERTAINTY
            if testing_params['uncertainty']['applied'] and testing_params['uncertainty']['epistemic']:
                for m in model.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()

            # RUN MODEL
            if model_params["name"] == "HeMISUnet" or \
                    ('film_layers' in model_params and any(model_params['film_layers'])):
                metadata = get_metadata(batch["input_metadata"], model_params)
                preds = model(input_samples, metadata)
            else:
                preds = model(input_samples)

        if model_params["name"] == "HeMISUnet":
            # Reconstruct image with only one modality
            input_samples = batch['input'][0]

        if model_params["name"] == "Modified3DUNet" and model_params["attention"] and ofolder:
            imed_visualize.save_feature_map(batch, "attentionblock2", os.path.dirname(ofolder), model, input_samples,
                                            slice_axis=test_loader.dataset.slice_axis)

        # PREDS TO CPU
        preds_cpu = preds.cpu()

        task = imed_utils.get_task(model_params["name"])
        if task == "classification":
            gt_npy_list.append(gt_samples.cpu().numpy())
            preds_npy_list.append(preds_cpu.data.numpy())

        # RECONSTRUCT 3D IMAGE
        last_batch_bool = (i == len(test_loader) - 1)

        slice_axis = imed_utils.AXIS_DCT[testing_params['slice_axis']]

        # LOOP ACROSS SAMPLES
        for smp_idx in range(len(preds_cpu)):
            if "bounding_box" in batch['input_metadata'][smp_idx][0]:
                imed_obj_detect.adjust_undo_transforms(testing_params["undo_transforms"].transforms, batch, smp_idx)

            if model_params["is_2d"]:
                last_sample_bool = (last_batch_bool and smp_idx == len(preds_cpu) - 1)
                # undo transformations
                preds_idx_undo, metadata_idx = testing_params["undo_transforms"](preds_cpu[smp_idx],
                                                                                 batch['gt_metadata'][smp_idx],
                                                                                 data_type='gt')
                # preds_idx_undo is a list n_label arrays
                preds_idx_arr = np.array(preds_idx_undo)

                # TODO: gt_filenames should not be a list
                fname_ref = metadata_idx[0]['gt_filenames'][0]

                # NEW COMPLETE VOLUME
                if pred_tmp_lst and (fname_ref != fname_tmp or last_sample_bool) and task != "classification":
                    # save the completely processed file as a nifti file
                    if ofolder:
                        fname_pred = os.path.join(ofolder, fname_tmp.split('/')[-1])
                        fname_pred = fname_pred.rsplit(testing_params['target_suffix'][0],1)[0] + '_pred.nii.gz'
                        # If Uncertainty running, then we save each simulation result
                        if testing_params['uncertainty']['applied']:
                            fname_pred = fname_pred.split('.nii.gz')[0] + '_' + str(i_monte_carlo).zfill(2) + '.nii.gz'
                            postprocessing = None
                    else:
                        fname_pred = None
                    output_nii = imed_inference.pred_to_nib(data_lst=pred_tmp_lst,
                                                        z_lst=z_tmp_lst,
                                                        fname_ref=fname_tmp,
                                                        fname_out=fname_pred,
                                                        slice_axis=slice_axis,
                                                        kernel_dim='2d',
                                                        bin_thr=-1,
                                                        postprocessing=postprocessing)
                    output_data = output_nii.get_fdata().transpose(3, 0, 1, 2)
                    preds_npy_list.append(output_data)

                    gt = get_gt(filenames)
                    gt_npy_list.append(gt)

                    output_nii_shape = output_nii.get_fdata().shape
                    if len(output_nii_shape) == 4 and output_nii_shape[-1] > 1 and ofolder:
                        imed_visualize.save_color_labels(np.stack(pred_tmp_lst, -1),
                                                     False,
                                                     fname_tmp,
                                                     fname_pred.split(".nii.gz")[0] + '_color.nii.gz',
                                                     imed_utils.AXIS_DCT[testing_params['slice_axis']])

                    # re-init pred_stack_lst
                    pred_tmp_lst, z_tmp_lst = [], []

                # add new sample to pred_tmp_lst, of size n_label X h X w ...
                pred_tmp_lst.append(preds_idx_arr)

                # TODO: slice_index should be stored in gt_metadata as well
                z_tmp_lst.append(int(batch['input_metadata'][smp_idx][0]['slice_index']))
                fname_tmp = fname_ref
                filenames = metadata_idx[0]['gt_filenames']

            else:
                pred_undo, metadata, last_sample_bool, volume, weight_matrix = \
                    imed_inference.volume_reconstruction(batch,
                                                     preds_cpu,
                                                     testing_params['undo_transforms'],
                                                     smp_idx, volume, weight_matrix)
                fname_ref = metadata[0]['gt_filenames'][0]
                # Indicator of last batch
                if last_sample_bool:
                    pred_undo = np.array(pred_undo)
                    if ofolder:
                        fname_pred = os.path.join(ofolder, fname_ref.split('/')[-1])
                        fname_pred = fname_pred.split(testing_params['target_suffix'][0])[0] + '_pred.nii.gz'
                        # If uncertainty running, then we save each simulation result
                        if testing_params['uncertainty']['applied']:
                            fname_pred = fname_pred.split('.nii.gz')[0] + '_' + str(i_monte_carlo).zfill(2) + '.nii.gz'
                    else:
                        fname_pred = None
                    # Choose only one modality
                    output_nii = imed_inference.pred_to_nib(data_lst=[pred_undo],
                                                        z_lst=[],
                                                        fname_ref=fname_ref,
                                                        fname_out=fname_pred,
                                                        slice_axis=slice_axis,
                                                        kernel_dim='3d',
                                                        bin_thr=-1,
                                                        postprocessing=postprocessing)
                    output_data = output_nii.get_fdata().transpose(3, 0, 1, 2)
                    preds_npy_list.append(output_data)

                    gt = get_gt(metadata[0]['gt_filenames'])
                    gt_npy_list.append(gt)
                    # Save merged labels with color

                    if pred_undo.shape[0] > 1 and ofolder:
                        imed_visualize.save_color_labels(pred_undo,
                                                     False,
                                                     batch['input_metadata'][smp_idx][0]['input_filenames'],
                                                     fname_pred.split(".nii.gz")[0] + '_color.nii.gz',
                                                     slice_axis)

    return preds_npy_list, gt_npy_list


def threshold_analysis(model_path, ds_lst, model_params, testing_params, metric="dice", increment=0.1,
                       fname_out="thr.png", cuda_available=True):
    """Run a threshold analysis to find the optimal threshold on a sub-dataset.

    Args:
        model_path (str): Model path.
        ds_lst (list): List of loaders.
        model_params (dict): Model's parameters.
        testing_params (dict): Testing parameters
        metric (str): Choice between "dice" and "recall_specificity". If "recall_specificity", then a ROC analysis
            is performed.
        increment (float): Increment between tested thresholds.
        fname_out (str): Plot output filename.
        cuda_available (bool): If True, CUDA is available.

    Returns:
        float: optimal threshold.
    """
    if metric not in ["dice", "recall_specificity"]:
        raise ValueError('\nChoice of metric for threshold analysis: dice, recall_specificity.')

    # Adjust some testing parameters
    testing_params["uncertainty"]["applied"] = False

    # Load model
    model = torch.load(model_path)
    # Eval mode
    model.eval()

    # List of thresholds
    thr_list = list(np.arange(0.0, 1.0, increment))[1:]

    # Init metric manager for each thr
    metric_fns = [imed_metrics.recall_score,
                  imed_metrics.dice_score,
                  imed_metrics.specificity_score]
    metric_dict = {thr: imed_metrics.MetricManager(metric_fns) for thr in thr_list}

    # Load
    loader = DataLoader(ConcatDataset(ds_lst), batch_size=testing_params["batch_size"],
                        shuffle=False, pin_memory=True, sampler=None,
                        collate_fn=imed_loader_utils.imed_collate,
                        num_workers=0)

    # Run inference
    preds_npy, gt_npy = run_inference(loader, model, model_params,
                                      testing_params,
                                      ofolder=None,
                                      cuda_available=cuda_available)

    print('\nRunning threshold analysis to find optimal threshold')
    # Make sure the GT is binarized
    gt_npy = [threshold_predictions(gt, thr=0.5) for gt in gt_npy]
    # Move threshold
    for thr in tqdm(thr_list, desc="Search"):
        preds_thr = [threshold_predictions(copy.deepcopy(pred), thr=thr) for pred in preds_npy]
        metric_dict[thr](preds_thr, gt_npy)

    # Get results
    tpr_list, fpr_list, dice_list = [], [], []
    for thr in thr_list:
        result_thr = metric_dict[thr].get_results()
        tpr_list.append(result_thr["recall_score"])
        fpr_list.append(1 - result_thr["specificity_score"])
        dice_list.append(result_thr["dice_score"])

    # Get optimal threshold
    if metric == "dice":
        diff_list = dice_list
    else:
        diff_list = [tpr - fpr for tpr, fpr in zip(tpr_list, fpr_list)]

    optimal_idx = np.max(np.where(diff_list == np.max(diff_list)))
    optimal_threshold = thr_list[optimal_idx]
    print('\tOptimal threshold: {}'.format(optimal_threshold))

    # Save plot
    print('\tSaving plot: {}'.format(fname_out))
    if metric == "dice":
        # Run plot
        imed_metrics.plot_dice_thr(thr_list, dice_list, optimal_idx, fname_out)
    else:
        # Add 0 and 1 as extrema
        tpr_list = [0.0] + tpr_list + [1.0]
        fpr_list = [0.0] + fpr_list + [1.0]
        optimal_idx += 1
        # Run plot
        imed_metrics.plot_roc_curve(tpr_list, fpr_list, optimal_idx, fname_out)

    return optimal_threshold


def get_gt(filenames):
    """Get ground truth data as numpy array.
    
    Args:
        filenames (list): List of ground truth filenames, one per class.
    Returns:
        ndarray: 4D numpy array.
    """
    gt_lst = []
    for gt in filenames:
        # For multi-label, if all labels are not in every image
        if gt is not None:
            gt_lst.append(nib.load(gt).get_fdata())
        else:
            gt_lst.append(np.zeros(gt_lst[0].shape))
    return np.array(gt_lst)

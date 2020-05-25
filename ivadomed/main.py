import json
import os
import random
import shutil
import sys
import time

import joblib
import nibabel as nib
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ivadomed import training as imed_training
from ivadomed import losses as imed_losses
from ivadomed import metrics as imed_metrics
from ivadomed import models as imed_models
from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader, film as imed_film

cudnn.benchmark = True


def cmd_test(context):
    ##### DEFINE DEVICE #####
    device = torch.device("cuda:" + str(context['gpu']) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_number = int(context["gpu"])
        torch.cuda.set_device(gpu_number)
        print("using GPU number {}".format(gpu_number))
    HeMIS = context['HeMIS']
    # Boolean which determines if the selected architecture is FiLMedUnet or Unet
    film_bool = bool(sum(context["film_layers"]))
    print('\nArchitecture: {}\n'.format('FiLMedUnet' if film_bool else 'Unet'))
    if context["metadata"] == "mri_params":
        print('\tInclude subjects with acquisition metadata available only.\n')
    else:
        print('\tInclude all subjects, with or without acquisition metadata.\n')

    # Aleatoric uncertainty
    if context['uncertainty']['aleatoric'] and context['uncertainty']['n_it'] > 0:
        transformation_dict = context["transformation_testing"]
    else:
        transformation_dict = context["transformation_validation"]

    # Compose Testing transforms
    val_transform = imed_transforms.Compose(transformation_dict, requires_undo=True)

    # inverse transformations
    val_undo_transform = imed_transforms.UndoCompose(val_transform)

    if context.get("split_path") is None:
        test_lst = joblib.load("./" + context["log_directory"] + "/split_datasets.joblib")['test']
    else:
        test_lst = joblib.load(context["split_path"])['test']

    ds_test = imed_loader.load_dataset(test_lst, val_transform, context)

    # if ROICrop2D in transform, then apply SliceFilter to ROI slices
    if 'ROICrop2D' in context["transformation_validation"].keys():
        ds_test = imed_loader_utils.filter_roi(ds_test, nb_nonzero_thr=context["slice_filter_roi"])

    if film_bool:  # normalize metadata before sending to network
        metadata_clustering_models = joblib.load(
            "./" + context["log_directory"] + "/clustering_models.joblib")

        ds_test = imed_film.normalize_metadata(ds_test,
                                               metadata_clustering_models,
                                               context["debugging"],
                                               context["metadata"],
                                               False)

        one_hot_encoder = joblib.load("./" + context["log_directory"] + "/one_hot_encoder.joblib")

    if not context["unet_3D"]:
        print(f"\nLoaded {len(ds_test)} {context['slice_axis']} slices for the test set.")
    else:
        print(f"\nLoaded {len(ds_test)} volumes of size {context['length_3D']} for the test set.")

    test_loader = DataLoader(ds_test, batch_size=context["batch_size"],
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=0)

    model = torch.load("./" + context["log_directory"] + "/best_model.pt", map_location=device)

    if cuda_available:
        model.cuda()
    model.eval()

    # create output folder for 3D prediction masks
    path_3Dpred = os.path.join(context['log_directory'], 'pred_masks')
    if not os.path.isdir(path_3Dpred):
        os.makedirs(path_3Dpred)

    metric_fns = [imed_metrics.dice_score,
                  imed_metrics.multi_class_dice_score,
                  imed_metrics.hausdorff_3D_score,
                  imed_metrics.precision_score,
                  imed_metrics.recall_score,
                  imed_metrics.specificity_score,
                  imed_metrics.intersection_over_union,
                  imed_metrics.accuracy_score]

    metric_mgr = imed_metrics.MetricManager(metric_fns)

    # number of Monte Carlo simulation
    if (context['uncertainty']['epistemic'] or context['uncertainty']['aleatoric']) and \
            context['uncertainty']['n_it'] > 0:
        n_monteCarlo = context['uncertainty']['n_it']
    else:
        n_monteCarlo = 1

    # Epistemic uncertainty
    if context['uncertainty']['epistemic'] and context['uncertainty']['n_it'] > 0:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    for i_monteCarlo in range(n_monteCarlo):
        pred_tmp_lst, z_tmp_lst, fname_tmp = [], [], ''
        for i, batch in enumerate(test_loader):
            # input_samples: list of n_batch tensors, whose size is n_channels X height X width X depth
            # gt_samples: idem with n_labels
            # batch['*_metadata']: list of n_batch lists, whose size is n_channels or n_labels
            input_samples, gt_samples = batch["input"] if not HeMIS \
                                            else imed_utils.unstack_tensors(batch["input"]), batch["gt"]

            with torch.no_grad():
                if cuda_available:
                    test_input = imed_utils.cuda(input_samples)
                    test_gt = imed_utils.cuda(gt_samples, non_blocking=True)

                else:
                    test_input = input_samples
                    test_gt = gt_samples

                # Epistemic uncertainty
                if context['uncertainty']['epistemic'] and context['uncertainty']['n_it'] > 0:
                    for m in model.modules():
                        if m.__class__.__name__.startswith('Dropout'):
                            m.train()

                if film_bool:
                    sample_metadata = batch["input_metadata"]
                    test_contrast = [sample_metadata[0][k]['contrast']
                                     for k in range(len(sample_metadata[0]))]

                    test_metadata = [one_hot_encoder.transform([sample_metadata[0][k]["film_input"]]).tolist()[0]
                                     for k in range(len(sample_metadata[0]))]

                    # Input the metadata related to the input samples
                    preds = model(test_input, test_metadata)
                elif HeMIS:
                    # TODO: @Andreanne: to modify?
                    missing_mod = batch["Missing_mod"]
                    preds = model(test_input, missing_mod)

                    # Reconstruct image with only one modality
                    batch['input'] = batch['input'][0]
                    batch['input_metadata'] = batch['input_metadata'][0]

                else:
                    preds = model(test_input)
                    if context["attention_unet"]:
                        imed_utils.save_feature_map(batch, "attentionblock2", context, model, test_input,
                                                    imed_utils.AXIS_DCT[context["slice_axis"]])

            # Preds to CPU
            preds_cpu = preds.cpu()

            # reconstruct 3D image
            for smp_idx in range(len(preds_cpu)):
                # undo transformations
                preds_idx_undo, metadata_idx = val_undo_transform(preds_cpu[smp_idx], batch["gt_metadata"][smp_idx],
                                                                  data_type='gt')

                # preds_idx_undo is a list of length n_label of arrays
                preds_idx_arr = np.array(preds_idx_undo)

                # TODO: gt_filenames should not be a list
                fname_ref = metadata_idx[0]['gt_filenames'][0]

                if not context['unet_3D']:
                    if pred_tmp_lst and (fname_ref != fname_tmp or (
                            i == len(test_loader) - 1 and smp_idx == len(batch['gt']) - 1)):  # new processed file
                        # save the completely processed file as a nifti file
                        fname_pred = os.path.join(path_3Dpred, fname_tmp.split('/')[-1])
                        fname_pred = fname_pred.split(context['target_suffix'][0])[0] + '_pred.nii.gz'

                        # If MonteCarlo, then we save each simulation result
                        if n_monteCarlo > 1:
                            fname_pred = fname_pred.split('.nii.gz')[0] + '_' + str(i_monteCarlo).zfill(2) + '.nii.gz'

                        output_nii = imed_utils.pred_to_nib(data_lst=pred_tmp_lst,
                                                            z_lst=z_tmp_lst,
                                                            fname_ref=fname_tmp,
                                                            fname_out=fname_pred,
                                                            slice_axis=imed_utils.AXIS_DCT[context['slice_axis']],
                                                            kernel_dim='2d',
                                                            bin_thr=0.5 if context["binarize_prediction"] else -1)

                        output_nii_shape = output_nii.get_fdata().shape
                        if len(output_nii_shape) == 4 and output_nii_shape[0] > 1:
                            imed_utils.save_color_labels(output_nii.get_fdata(),
                                                         context["binarize_prediction"],
                                                         fname_tmp,
                                                         fname_pred.split(".nii.gz")[0] + '_color.nii.gz',
                                                         imed_utils.AXIS_DCT[context['slice_axis']])

                        # re-init pred_stack_lst
                        pred_tmp_lst, z_tmp_lst = [], []

                    # add new sample to pred_tmp_lst, of size n_label X h X w ...
                    pred_tmp_lst.append(preds_idx_arr)

                    # TODO: slice_index should be stored in gt_metadata as well
                    z_tmp_lst.append(int(batch['input_metadata'][smp_idx][0]['slice_index']))
                    fname_tmp = fname_ref

                else:
                    # TODO: Add reconstruction for subvolumes
                    fname_pred = os.path.join(path_3Dpred, fname_ref.split('/')[-1])
                    fname_pred = fname_pred.split(context['target_suffix'][0])[0] + '_pred.nii.gz'
                    # If MonteCarlo, then we save each simulation result
                    if n_monteCarlo > 1:
                        fname_pred = fname_pred.split('.nii.gz')[0] + '_' + str(i_monteCarlo).zfill(2) + '.nii.gz'

                    # Choose only one modality
                    imed_utils.pred_to_nib(data_lst=[preds_idx_arr],
                                           z_lst=[],
                                           fname_ref=fname_ref,
                                           fname_out=fname_pred,
                                           slice_axis=imed_utils.AXIS_DCT[context['slice_axis']],
                                           kernel_dim='3d',
                                           bin_thr=0.5 if context["binarize_prediction"] else -1)

                    # Save merged labels with color
                    if preds_idx_arr.shape[0] > 1:
                        imed_utils.save_color_labels(preds_idx_arr,
                                                     context['binarize_prediction'],
                                                     batch['input_metadata'][smp_idx][0]['input_filenames'],
                                                     fname_pred.split(".nii.gz")[0] + '_color.nii.gz',
                                                     imed_utils.AXIS_DCT[context['slice_axis']])

            # Metrics computation
            gt_npy = gt_samples.numpy().astype(np.uint8)

            preds_npy = preds.data.cpu().numpy()
            if context["binarize_prediction"]:
                preds_npy = imed_postpro.threshold_predictions(preds_npy)
            preds_npy = preds_npy.astype(np.uint8)

            metric_mgr(preds_npy, gt_npy)

    # COMPUTE UNCERTAINTY MAPS
    if (context['uncertainty']['epistemic'] or context['uncertainty']['aleatoric']) and \
            context['uncertainty']['n_it'] > 0:
        imed_utils.run_uncertainty(ifolder=path_3Dpred)

    metrics_dict = metric_mgr.get_results()
    metric_mgr.reset()
    print(metrics_dict)
    return metrics_dict


def cmd_eval(context):
    path_pred = os.path.join(context['log_directory'], 'pred_masks')
    if not os.path.isdir(path_pred):
        print('\nRun Inference\n')
        metrics_dict = cmd_test(context)
    print('\nRun Evaluation on {}\n'.format(path_pred))

    ##### DEFINE DEVICE #####
    device = torch.device("cpu")
    print("Working on {}.".format(device))

    # create output folder for results
    path_results = os.path.join(context['log_directory'], 'results_eval')
    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    # init data frame
    df_results = pd.DataFrame()

    # list subject_acquisition
    subj_acq_lst = [f.split('_pred')[0]
                    for f in os.listdir(path_pred) if f.endswith('_pred.nii.gz')]

    # loop across subj_acq
    for subj_acq in tqdm(subj_acq_lst, desc="Evaluation"):
        subj, acq = subj_acq.split('_')[0], '_'.join(subj_acq.split('_')[1:])

        fname_pred = os.path.join(path_pred, subj_acq + '_pred.nii.gz')
        fname_gt = []
        for suffix in context['target_suffix']:
            fname_gt.append(os.path.join(context['bids_path'], 'derivatives', 'labels', subj, 'anat',
                                         subj_acq + suffix + '.nii.gz'))

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
                                              params=context['eval_params'])

        # run eval
        results_pred, data_painted = eval.run_eval()
        # save painted data, TP FP FN
        fname_paint = fname_pred.split('.nii.gz')[0] + '_painted.nii.gz'
        nib_painted = nib.Nifti1Image(data_painted, nib_pred.affine)
        nib.save(nib_painted, fname_paint)

        # save results of this fname_pred
        results_pred['image_id'] = subj_acq
        df_results = df_results.append(results_pred, ignore_index=True)

    df_results = df_results.set_index('image_id')
    df_results.to_csv(os.path.join(path_results, 'evaluation_3Dmetrics.csv'))

    print(df_results.head(5))
    return metrics_dict, df_results


def define_device(gpu_id):
    """Define the device used for the process of interest.

    Args:
        gpu_id (int): ID of the GPU
    Returns:
        Bool: True if cuda is available
    """
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("Cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_number = int(gpu_id)
        torch.cuda.set_device(gpu_number)
        print("Using GPU number {}".format(gpu_number))
    return cuda_available


def get_new_subject_split(path_folder, center_test, split_method, random_seed,
                          train_frac, test_frac, log_directory):
    """Randomly split dataset between training / validation / testing.

    Randomly split dataset between training / validation / testing
        and save it in log_directory + "/split_datasets.joblib"
    Args:
        path_folder (string): Dataset folder
        center_test (list): list of centers to include in the testing set
        split_method (string): see imed_loader_utils.split_dataset
        random_seed (int):
        train_frac (float): between 0 and 1
        test_frac (float): between 0 and 1
        log_directory (string): output folder
    Returns:
        list, list list: Training, validation and testing subjects lists
    """
    train_lst, valid_lst, test_lst = imed_loader_utils.split_dataset(path_folder=path_folder,
                                                                     center_test_lst=center_test,
                                                                     split_method=split_method,
                                                                     random_seed=random_seed,
                                                                     train_frac=train_frac,
                                                                     test_frac=test_frac)

    # save the subject distribution
    split_dct = {'train': train_lst, 'valid': valid_lst, 'test': test_lst}
    joblib.dump(split_dct, "./" + log_directory + "/split_datasets.joblib")

    return train_lst, valid_lst, test_lst


def get_subdatasets_subjects_list(split_params, bids_path, log_directory):
    """Get lists of subjects for each sub-dataset between training / validation / testing.

    Args:
        split_params (dict):
        bids_path (string): Path to the BIDS dataset
        log_directory (string): output folder
    Returns:
        list, list list: Training, validation and testing subjects lists
    """
    if split_params["fname_split"]:
        # Load subjects lists
        old_split = joblib.load(split_params["fname_split"])
        train_lst, valid_lst, test_lst = old_split['train'], old_split['valid'], old_split['test']
    else:
        train_lst, valid_lst, test_lst = get_new_subject_split(path_folder=bids_path,
                                                               center_test=split_params['center_test'],
                                                               split_method=split_params['method'],
                                                               random_seed=split_params['random_seed'],
                                                               train_frac=split_params['train_fraction'],
                                                               test_frac=split_params['test_fraction'],
                                                               log_directory=log_directory)
    return train_lst, valid_lst, test_lst


def normalize_film_metadata(ds_train, ds_val, metadata_type, debugging):
    if metadata_type == "mri_params":
        metadata_vector = ["RepetitionTime", "EchoTime", "FlipAngle"]
        metadata_clustering_models = imed_film.clustering_fit(ds_train.metadata, metadata_vector)
    else:
        metadata_clustering_models = None

    ds_train, train_onehotencoder = imed_film.normalize_metadata(ds_train,
                                                                 metadata_clustering_models,
                                                                 debugging,
                                                                 metadata_type,
                                                                 True)
    ds_val = imed_film.normalize_metadata(ds_val,
                                          metadata_clustering_models,
                                          debugging,
                                          metadata_type)

    return ds_train, ds_val, train_onehotencoder


def display_selected_model_spec(params):
    """Display in terminal the selected model and its parameters.

    Args:
        params (dict): keys are param names and values are param values
    Returns:
        None
    """
    print('\nSelected architecture: {}, with the following parameters:'.format(params["name"]))
    for k in list(params.keys()):
        if k != "name":
            print('\t{}: {}'.format(k, params[k]))


def get_subdatasets_transforms(transform_params):
    """Get transformation parameters for each subdataset: training, validation and testing.

    Args:
        transform_params (dict):
    Returns:
        dict, dict, dict
    """
    train, valid, test = {}, {}, {}
    subdataset_default = ["training", "validation", "testing"]
    # Loop across transformations
    for transform_name in transform_params:
        subdataset_list = ["training", "validation", "testing"]
        # Only consider subdatasets listed in dataset_type
        if "dataset_type" in transform_params[transform_name]:
            subdataset_list = transform_params[transform_name]["dataset_type"]
        # Add current transformation to the relevant subdataset transformation dictionaries
        for subds_name, subds_dict in zip(subdataset_default, [train, valid, test]):
            if subds_name in subdataset_list:
                subds_dict.update({transform_params[transform_name]})
    return train, valid, test


def run_main():
    if len(sys.argv) <= 1:
        print("\nivadomed [config.json]\n")
        return

    with open(sys.argv[1], "r") as fhandle:
        context = json.load(fhandle)

    command = context["command"]
    log_directory = context["log_directory"]

    # Define device
    cuda_available = define_device(context['gpu'])

    # Get subject lists
    train_lst, valid_lst, test_lst = get_subdatasets_subjects_list(context["split_dataset"],
                                                                   context['bids_path'],
                                                                   log_directory)

    # Get transforms for each subdataset
    transform_train_params, transform_valid_params, transform_test_params = \
        get_subdatasets_transforms(context["transformation"])

    # Loader params
    loader_params = {"bids_path": context['bids_path'],
                     "target_suffix": context["target_suffix"],
                     "roi_params": context["roi"],
                     "contrast_params": context["contrasts"],
                     "slice_filter_params": context["slice_filter"],
                     "slice_axis": context["slice_axis"],
                     "multichannel": context["multichannel"],
                     "metadata_type": context["FiLM"]["metadata"]}

    if command == 'train':
        # PARSE PARAMETERS
        film_params = context["FiLM"] if context["FiLM"]["metadata"] != "without" else None
        multichannel_params = context["contrast"]["train_validation"] if context["multichannel"] else None
        mixup_params = float(context["mixup_alpha"]) if context["mixup_alpha"] else None
        # Disable some attributes
        if film_params:
            multichannel_params = None
            context["HeMIS"] = False
            mixup_params = False
        if multichannel_params:
            context["HeMIS"] = False

        # MODEL PARAMETERS
        model_available = ['unet_2D', 'unet3D', 'HeMIS']
        model_context_list = [model_name for model_name in model_available
                              if model_name in context and context[model_name]["applied"]]
        if len(model_context_list) == 1:
            model_name = model_context_list[0]
            model_params = context[model_name]
        elif len(model_context_list) > 1:
            print('ERROR: Several models are selected in the configuration file: {}.'
                  'Please select only one.'.format(model_context_list))
            exit()
        elif film_params:
            model_name = 'FiLMedUnet_2D'
            model_params = film_params
        else:
            # Select default model
            model_name = 'unet_2D'
            model_params = {}
        # Update params
        model_params.update({"name": model_name,
                             "depth": context['depth'],
                             "multichannel": multichannel_params,
                             "n_out_channel": context["out_channel"]})
        display_selected_model_spec(params=model_params)

        # LOAD DATASET
        # Update loader params
        loader_params.update({"model_params": model_params})
        # Get Training dataset
        ds_train = imed_loader.load_dataset(**{**loader_params,
                                               **{'data_list': train_lst, 'transforms_params': transform_train_params,
                                                  'dataset_type': 'training'}})
        # Get Validation dataset
        ds_valid = imed_loader.load_dataset(**{**loader_params,
                                               **{'data_list': valid_lst, 'transforms_params': transform_valid_params,
                                                  'dataset_type': 'validation'}})
        # If FiLM, normalize data
        if film_params:
            # Normalize metadata before sending to the FiLM network
            ds_train, ds_valid, train_onehotencoder = normalize_film_metadata(ds_train=ds_train,
                                                                              ds_val=ds_valid,
                                                                              metadata_type=film_params['metadata'],
                                                                              debugging=context["debugging"])
            film_params.update({"film_onehotencoder": train_onehotencoder})

        # RUN TRAINING
        imed_training.train(model_params=model_params,
                            dataset_train=ds_train,
                            dataset_val=ds_valid,
                            log_directory=log_directory,
                            cuda_available=cuda_available,
                            balance_samples=context["balance_samples"],
                            mixup_params=mixup_params)

        # Save config file within log_directory
        shutil.copyfile(sys.argv[1], "./" + log_directory + "/config_file.json")

    elif command == 'test':
        cmd_test(context)
    elif command == 'eval':
        cmd_eval(context)


if __name__ == "__main__":
    run_main()

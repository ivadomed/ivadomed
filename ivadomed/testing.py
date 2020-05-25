import time
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ivadomed import losses as imed_losses
from ivadomed import metrics as imed_metrics
from ivadomed import models as imed_models
from ivadomed import postprocessing as imed_postpro
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader


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

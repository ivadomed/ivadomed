import sys
import json
import os
import time
import shutil
import random
import joblib
from math import exp
import numpy as np

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from medicaltorch.filters import SliceFilter
from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms

from tqdm import tqdm

from ivadomed import loader as loader
from ivadomed import models
from ivadomed import losses
from ivadomed.utils import *
import ivadomed.transforms as ivadomed_transforms

cudnn.benchmark = True


def cmd_train(context):
    """Main command to train the network.

    :param context: this is a dictionary with all data from the
                    configuration file
    """
    ##### DEFINE DEVICE #####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("Cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_number = int(context["gpu"])
        torch.cuda.set_device(gpu_number)
        print("Using GPU number {}".format(gpu_number))

    # Boolean which determines if the selected architecture is FiLMedUnet or Unet or MixupUnet
    metadata_bool = False if context["metadata"] == "without" else True
    film_bool = (bool(sum(context["film_layers"])) and metadata_bool)
    HeMIS = context['missing_modality']
    if film_bool:
        context["multichannel"] = False

    if bool(sum(context["film_layers"])) and not (metadata_bool):
        print('\tWarning FiLM disabled since metadata is disabled')
    print('\nArchitecture: {} with a depth of {}.\n' \
          .format('FiLMedUnet' if film_bool else 'HeMIS-Unet' if HeMIS else 'Unet', context['depth']))

    mixup_bool = False if film_bool else bool(context["mixup_bool"])
    mixup_alpha = float(context["mixup_alpha"])
    if not film_bool and mixup_bool:
        print('\twith Mixup (alpha={})\n'.format(mixup_alpha))
    if context["metadata"] == "mri_params":
        print('\tInclude subjects with acquisition metadata available only.\n')
    else:
        print('\tInclude all subjects, with or without acquisition metadata.\n')

    # Write the metrics, images, etc to TensorBoard format
    writer = SummaryWriter(log_dir=context["log_directory"])

    # These are the training transformations
    training_transform_list = []
    for transform in context["transformation_training"].keys():
        parameters = context["transformation_training"][transform]
        if transform in ivadomed_transforms.get_transform_names():
            training_transform_list.append(getattr(ivadomed_transforms, transform)(**parameters))
        else:
            training_transform_list.append(getattr(mt_transforms, transform)(**parameters))

    train_transform = transforms.Compose(training_transform_list)

    # These are the validation/testing transformations
    validation_transform_list = []
    for transform in context["transformation_validation"].keys():
        parameters = context["transformation_validation"][transform]
        if transform in ivadomed_transforms.get_transform_names():
            validation_transform_list.append(getattr(ivadomed_transforms, transform)(**parameters))
        else:
            validation_transform_list.append(getattr(mt_transforms, transform)(**parameters))

    val_transform = transforms.Compose(validation_transform_list)

    # Randomly split dataset between training / validation / testing
    if context.get("split_path") is None:
        train_lst, valid_lst, test_lst = loader.split_dataset(path_folder=context["bids_path"],
                                                              center_test_lst=context["center_test"],
                                                              split_method=context["split_method"],
                                                              random_seed=context["random_seed"],
                                                              train_frac=context["train_fraction"],
                                                              test_frac=context["test_fraction"])

        # save the subject distribution
        split_dct = {'train': train_lst, 'valid': valid_lst, 'test': test_lst}
        joblib.dump(split_dct, "./" + context["log_directory"] + "/split_datasets.joblib")

    else:
        train_lst = joblib.load(context["split_path"])['train']
        valid_lst = joblib.load(context["split_path"])['valid']

    axis_dct = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    # This code will iterate over the folders and load the data, filtering
    # the slices without labels and then concatenating all the datasets together
    ds_train = loader.BidsDataset(context["bids_path"],
                                  subject_lst=train_lst,
                                  target_suffix=context["target_suffix"],
                                  roi_suffix=context["roi_suffix"],
                                  contrast_lst=context["contrast_train_validation"],
                                  metadata_choice=context["metadata"],
                                  contrast_balance=context["contrast_balance"],
                                  slice_axis=axis_dct[context["slice_axis"]],
                                  transform=train_transform,
                                  multichannel=context['multichannel'],
                                  slice_filter_fn=SliceFilter(**context["slice_filter"]))

    # if ROICrop2D in transform, then apply SliceFilter to ROI slices
    if 'ROICrop2D' in context["transformation_training"].keys():
        ds_train.filter_roi(nb_nonzero_thr=context["slice_filter_roi"])

    if film_bool:  # normalize metadata before sending to the network
        if context["metadata"] == "mri_params":
            metadata_vector = ["RepetitionTime", "EchoTime", "FlipAngle"]
            metadata_clustering_models = loader.clustering_fit(ds_train.metadata, metadata_vector)
        else:
            metadata_clustering_models = None
        ds_train, train_onehotencoder = loader.normalize_metadata(ds_train,
                                                                  metadata_clustering_models,
                                                                  context["debugging"],
                                                                  context["metadata"],
                                                                  True)

    print(f"Loaded {len(ds_train)} {context['slice_axis']} slices for the training set.")

    if context['balance_samples']:
        sampler_train = loader.BalancedSampler(ds_train)
        shuffle_train = False
    else:
        sampler_train, shuffle_train = None, True

    train_loader = DataLoader(ds_train, batch_size=context["batch_size"],
                              shuffle=shuffle_train, pin_memory=True, sampler=sampler_train,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=0)

    # Validation dataset ------------------------------------------------------
    ds_val = loader.BidsDataset(context["bids_path"],
                                subject_lst=valid_lst,
                                target_suffix=context["target_suffix"],
                                roi_suffix=context["roi_suffix"],
                                contrast_lst=context["contrast_train_validation"],
                                metadata_choice=context["metadata"],
                                contrast_balance=context["contrast_balance"],
                                slice_axis=axis_dct[context["slice_axis"]],
                                transform=val_transform,
                                multichannel=context['multichannel'],

                                slice_filter_fn=SliceFilter(**context["slice_filter"]))

    # if ROICrop2D in transform, then apply SliceFilter to ROI slices
    if 'ROICrop2D' in context["transformation_validation"].keys():
        ds_val.filter_roi(nb_nonzero_thr=context["slice_filter_roi"])

    if film_bool:  # normalize metadata before sending to network
        ds_val = loader.normalize_metadata(ds_val,
                                           metadata_clustering_models,
                                           context["debugging"],
                                           context["metadata"],
                                           False)

    print(f"Loaded {len(ds_val)} {context['slice_axis']} slices for the validation set.")

    if context['balance_samples']:
        sampler_val = loader.BalancedSampler(ds_val)
        shuffle_val = False
    else:
        sampler_val, shuffle_val = None, True

    val_loader = DataLoader(ds_val, batch_size=context["batch_size"],
                            shuffle=shuffle_val, pin_memory=True, sampler=sampler_val,
                            collate_fn=mt_datasets.mt_collate,
                            num_workers=0)

    if film_bool:
        # Modulated U-net model with FiLM layers
        model = models.FiLMedUnet(n_metadata=len([ll for l in train_onehotencoder.categories_ for ll in l]),
                                  film_bool=context["film_layers"],
                                  drop_rate=context["dropout_rate"],
                                  bn_momentum=context["batch_norm_momentum"])
    else:
        # Traditional U-Net model
        in_channel = 1
        if context['multichannel']:
            in_channel = len(context['contrast_train_validation'])

        model = models.Unet(in_channel=in_channel,
                            out_channel=context['out_channel'],
                            depth=context['depth'],
                            drop_rate=context["dropout_rate"],
                            bn_momentum=context["batch_norm_momentum"])

    if cuda_available:
        model.cuda()

    num_epochs = context["num_epochs"]
    initial_lr = context["initial_lr"]

    # Using Adam
    step_scheduler_batch = False
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    if context["lr_scheduler"]["name"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif context["lr_scheduler"]["name"] == "CosineAnnealingWarmRestarts":
        T_0 = context["lr_scheduler"]["T_0"]
        T_mult = context["lr_scheduler"]["T_mult"]
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    elif context["lr_scheduler"]["name"] == "CyclicLR":
        base_lr, max_lr = context["lr_scheduler"]["base_lr"], context["lr_scheduler"]["max_lr"]
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, mode="triangular2", cycle_momentum=False)
        step_scheduler_batch = True
    else:
        print(
            "Unknown LR Scheduler name, please choose between 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', or 'CyclicLR'")
        exit()

    # Create dict containing gammas and betas after each FiLM layer.
    gammas_dict = {i: [] for i in range(1, 9)}
    betas_dict = {i: [] for i in range(1, 9)}

    # Create a list containing the contrast of all batch images
    var_contrast_list = []

    # Loss
    if context["loss"]["name"] in ["dice", "cross_entropy", "focal", "gdl", "focal_dice"]:
        if context["loss"]["name"] == "cross_entropy":
            loss_fct = nn.BCELoss()

        elif context["loss"]["name"] == "focal":
            loss_fct = losses.FocalLoss(gamma=context["loss"]["params"]["gamma"],
                                        alpha=context["loss"]["params"]["alpha"])
            print("\nLoss function: {}, with gamma={}, alpha={}.\n".format(context["loss"]["name"],
                                                                           context["loss"]["params"]["gamma"],
                                                                           context["loss"]["params"]["alpha"]))
        elif context["loss"]["name"] == "gdl":
            loss_fct = losses.GeneralizedDiceLoss()

        elif context["loss"]["name"] == "focal_dice":
            loss_fct = losses.FocalDiceLoss(beta=context["loss"]["params"]["beta"],
                                            gamma=context["loss"]["params"]["gamma"],
                                            alpha=context["loss"]["params"]["alpha"])
            print("\nLoss function: {}, with beta={}, gamma={} and alpha={}.\n".format(context["loss"]["name"],
                                                                                       context["loss"]["params"][
                                                                                           "beta"],
                                                                                       context["loss"]["params"][
                                                                                           "gamma"],
                                                                                       context["loss"]["params"][
                                                                                           "alpha"]))

        if not context["loss"]["name"].startswith("focal"):
            print("\nLoss function: {}.\n".format(context["loss"]["name"]))

    else:
        print("Unknown Loss function, please choose between 'dice', 'focal', 'focal_dice', 'gdl' or 'cross_entropy'")
        exit()

    # Training loop -----------------------------------------------------------

    best_training_dice, best_training_loss, best_validation_loss, best_validation_dice = float("inf"), float(
        "inf"), float("inf"), float("inf")

    patience = context["early_stopping_patience"]
    patience_count = 0
    epsilon = context["early_stopping_epsilon"]
    val_losses = []

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        start_time = time.time()

        lr = scheduler.get_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)

        model.train()
        train_loss_total, dice_train_loss_total = 0.0, 0.0

        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples = batch["input"], batch["gt"]

            # mixup data
            if mixup_bool and not film_bool:
                input_samples, gt_samples, lambda_tensor = mixup(input_samples, gt_samples, mixup_alpha)

                # if debugging and first epoch, then save samples as png in ofolder
                if context["debugging"] and epoch == 1 and random.random() < 0.1:
                    mixup_folder = os.path.join(context["log_directory"], 'mixup')
                    if not os.path.isdir(mixup_folder):
                        os.makedirs(mixup_folder)
                    random_idx = np.random.randint(0, input_samples.size()[0])
                    val_gt = np.unique(gt_samples.data.numpy()[random_idx, 0, :, :])
                    mixup_fname_pref = os.path.join(mixup_folder, str(i).zfill(3) + '_' + str(
                        lambda_tensor.data.numpy()[0]) + '_' + str(random_idx).zfill(3) + '.png')
                    save_mixup_sample(input_samples.data.numpy()[random_idx, 0, :, :],
                                      gt_samples.data.numpy()[random_idx, 0, :, :],
                                      mixup_fname_pref)

            # The variable sample_metadata is where the MRI physics parameters are

            if cuda_available:
                var_input = input_samples.cuda()
                var_gt = gt_samples.cuda(non_blocking=True)
            else:
                var_input = input_samples
                var_gt = gt_samples

            if film_bool:
                # var_contrast is the list of the batch sample's contrasts (eg T2w, T1w).
                sample_metadata = batch["input_metadata"]
                var_contrast = [sample_metadata[k]['contrast'] for k in range(len(sample_metadata))]

                var_metadata = [train_onehotencoder.transform([sample_metadata[k]['film_input']]).tolist()[0] for k in
                                range(len(sample_metadata))]
                preds = model(var_input, var_metadata)  # Input the metadata related to the input samples
            else:
                preds = model(var_input)

            if context["loss"]["name"] == "dice":
                loss = - losses.dice_loss(preds, var_gt)
            else:
                loss = loss_fct(preds, var_gt)
                dice_train_loss_total += losses.dice_loss(preds, var_gt).item()
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if step_scheduler_batch:
                scheduler.step()

            num_steps += 1

            # Only write sample at the first step
            if i == 0:
                grid_img = vutils.make_grid(input_samples,
                                            normalize=True,
                                            scale_each=True)
                writer.add_image('Train/Input', grid_img, epoch)

                grid_img = vutils.make_grid(preds.data.cpu(),
                                            normalize=True,
                                            scale_each=True)
                writer.add_image('Train/Predictions', grid_img, epoch)

                grid_img = vutils.make_grid(gt_samples,
                                            normalize=True,
                                            scale_each=True)
                writer.add_image('Train/Ground Truth', grid_img, epoch)

        train_loss_total_avg = train_loss_total / num_steps
        if not step_scheduler_batch:
            scheduler.step()

        tqdm.write(f"Epoch {epoch} training loss: {train_loss_total_avg:.4f}.")
        if context["loss"]["name"] != 'dice':
            dice_train_loss_total_avg = dice_train_loss_total / num_steps
            tqdm.write(f"\tDice training loss: {dice_train_loss_total_avg:.4f}.")

        # Validation loop -----------------------------------------------------
        model.eval()
        val_loss_total, dice_val_loss_total = 0.0, 0.0
        num_steps = 0

        metric_fns = [dice_score,  # from ivadomed/utils.py
                      mt_metrics.hausdorff_score,
                      mt_metrics.precision_score,
                      mt_metrics.recall_score,
                      mt_metrics.specificity_score,
                      mt_metrics.intersection_over_union,
                      mt_metrics.accuracy_score]

        metric_mgr = IvadoMetricManager(metric_fns)

        for i, batch in enumerate(val_loader):
            input_samples, gt_samples = batch["input"], batch["gt"]

            with torch.no_grad():
                if cuda_available:
                    var_input = input_samples.cuda()
                    var_gt = gt_samples.cuda(non_blocking=True)
                else:
                    var_input = input_samples
                    var_gt = gt_samples

                if film_bool:
                    sample_metadata = batch["input_metadata"]
                    # var_contrast is the list of the batch sample's contrasts (eg T2w, T1w).
                    var_contrast = [sample_metadata[k]['contrast'] for k in range(len(sample_metadata))]

                    var_metadata = [train_onehotencoder.transform([sample_metadata[k]['film_input']]).tolist()[0] for k
                                    in range(len(sample_metadata))]
                    preds = model(var_input, var_metadata)  # Input the metadata related to the input samples
                else:
                    preds = model(var_input)

                if context["loss"]["name"] == "dice":
                    loss = - losses.dice_loss(preds, var_gt)
                else:
                    loss = loss_fct(preds, var_gt)
                    dice_val_loss_total += losses.dice_loss(preds, var_gt).item()
                val_loss_total += loss.item()

            # Metrics computation
            gt_npy = gt_samples.numpy().astype(np.uint8)
            gt_npy = gt_npy.squeeze(axis=1)

            preds_npy = preds.data.cpu().numpy()
            preds_npy = threshold_predictions(preds_npy)
            preds_npy = preds_npy.astype(np.uint8)
            preds_npy = preds_npy.squeeze(axis=1)

            metric_mgr(preds_npy, gt_npy)

            num_steps += 1

            # Only write sample at the first step
            if i == 0:
                grid_img = vutils.make_grid(input_samples,
                                            normalize=True,
                                            scale_each=True)
                writer.add_image('Validation/Input', grid_img, epoch)

                grid_img = vutils.make_grid(preds.data.cpu(),
                                            normalize=True,
                                            scale_each=True)
                writer.add_image('Validation/Predictions', grid_img, epoch)

                grid_img = vutils.make_grid(gt_samples,
                                            normalize=True,
                                            scale_each=True)
                writer.add_image('Validation/Ground Truth', grid_img, epoch)

            # Store the values of gammas and betas after the last epoch for each batch
            if film_bool and epoch == num_epochs and i < int(len(ds_val) / context["batch_size"]) + 1:

                # Get all the contrasts of all batches
                var_contrast_list.append(var_contrast)

                # Get the list containing the number of film layers
                film_layers = context["film_layers"]

                # Fill the lists of gammas and betas
                for idx in [i for i, x in enumerate(film_layers) if x]:
                    attr_stg = 'film' + str(idx)
                    layer_cur = getattr(model, attr_stg)
                    gammas_dict[idx + 1].append(layer_cur.gammas[:, :, 0, 0].cpu().numpy())
                    betas_dict[idx + 1].append(layer_cur.betas[:, :, 0, 0].cpu().numpy())

        metrics_dict = metric_mgr.get_results()
        metric_mgr.reset()

        writer.add_scalars('Validation/Metrics', metrics_dict, epoch)
        val_loss_total_avg = val_loss_total / num_steps
        writer.add_scalars('losses', {
            'train_loss': train_loss_total_avg,
            'val_loss': val_loss_total_avg,
        }, epoch)

        tqdm.write(f"Epoch {epoch} validation loss: {val_loss_total_avg:.4f}.")
        if context["loss"]["name"] != 'dice':
            dice_val_loss_total_avg = dice_val_loss_total / num_steps
            tqdm.write(f"\tDice validation loss: {dice_val_loss_total_avg:.4f}.")

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

        if val_loss_total_avg < best_validation_loss:
            best_validation_loss = val_loss_total_avg
            best_training_loss = train_loss_total_avg

            if context["loss"]["name"] != 'dice':
                best_validation_dice = dice_val_loss_total_avg
                best_training_dice = dice_train_loss_total_avg
            else:
                best_validation_dice = best_validation_loss
                best_training_dice = best_training_loss
            torch.save(model, "./" + context["log_directory"] + "/best_model.pt")

        # Early stopping : break if val loss doesn't improve by at least epsilon percent for N=patience epochs
        val_losses.append(val_loss_total_avg)

        if epoch > 1:
            if (val_losses[-2] - val_losses[-1]) * 100 / abs(val_losses[-1]) < epsilon:
                patience_count += 1
        if patience_count >= patience:
            print(f"Stopping training due to {patience} epochs without improvements")
            break

    # Save final model
    torch.save(model, "./" + context["log_directory"] + "/final_model.pt")
    if film_bool:  # save clustering and OneHotEncoding models
        joblib.dump(metadata_clustering_models, "./" + context["log_directory"] + "/clustering_models.joblib")
        joblib.dump(train_onehotencoder, "./" + context["log_directory"] + "/one_hot_encoder.joblib")

        # Convert list of gammas/betas into numpy arrays
        gammas_dict = {i: np.array(gammas_dict[i]) for i in range(1, 9)}
        betas_dict = {i: np.array(betas_dict[i]) for i in range(1, 9)}

        # Save the numpy arrays for gammas/betas inside files.npy in log_directory
        for i in range(1, 9):
            np.save(context["log_directory"] + f"/gamma_layer_{i}.npy", gammas_dict[i])
            np.save(context["log_directory"] + f"/beta_layer_{i}.npy", betas_dict[i])

        # Convert into numpy and save the contrasts of all batch images
        contrast_images = np.array(var_contrast_list)
        np.save(context["log_directory"] + "/contrast_images.npy", contrast_images)

    writer.close()
    return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss


def cmd_test(context):
    ##### DEFINE DEVICE #####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_number = int(context["gpu"])
        torch.cuda.set_device(gpu_number)
        print("using GPU number {}".format(gpu_number))

    # Boolean which determines if the selected architecture is FiLMedUnet or Unet
    film_bool = bool(sum(context["film_layers"]))
    print('\nArchitecture: {}\n'.format('FiLMedUnet' if film_bool else 'Unet'))
    if context["metadata"] == "mri_params":
        print('\tInclude subjects with acquisition metadata available only.\n')
    else:
        print('\tInclude all subjects, with or without acquisition metadata.\n')

    # These are the validation/testing transformations
    validation_transform_list = []
    for transform in context["transformation_validation"].keys():
        parameters = context["transformation_validation"][transform]
        if transform in ivadomed_transforms.get_transform_names():
            validation_transform_list.append(getattr(ivadomed_transforms, transform)(**parameters))
        else:
            validation_transform_list.append(getattr(mt_transforms, transform)(**parameters))

    val_transform = transforms.Compose(validation_transform_list)

    if context.get("split_path") is None:
        test_lst = joblib.load("./" + context["log_directory"] + "/split_datasets.joblib")['test']
    else:
        test_lst = joblib.load(context["split_path"])['test']

    axis_dct = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    ds_test = loader.BidsDataset(context["bids_path"],
                                 subject_lst=test_lst,
                                 target_suffix=context["target_suffix"],
                                 roi_suffix=context["roi_suffix"],
                                 contrast_lst=context["contrast_test"],
                                 metadata_choice=context["metadata"],
                                 contrast_balance=context["contrast_balance"],
                                 slice_axis=axis_dct[context["slice_axis"]],
                                 transform=val_transform,
                                 slice_filter_fn=SliceFilter(**context["slice_filter"]),
                                 multichannel=context["multichannel"])

    # if ROICrop2D in transform, then apply SliceFilter to ROI slices
    if 'ROICrop2D' in context["transformation_validation"].keys():
        ds_test.filter_roi(nb_nonzero_thr=context["slice_filter_roi"])

    if film_bool:  # normalize metadata before sending to network
        metadata_clustering_models = joblib.load("./" + context["log_directory"] + "/clustering_models.joblib")
        ds_test = loader.normalize_metadata(ds_test,
                                            metadata_clustering_models,
                                            context["debugging"],
                                            context["metadata"],
                                            False)

        one_hot_encoder = joblib.load("./" + context["log_directory"] + "/one_hot_encoder.joblib")

    print(f"Loaded {len(ds_test)} {context['slice_axis']} slices for the test set.")
    test_loader = DataLoader(ds_test, batch_size=context["batch_size"],
                             shuffle=True, pin_memory=True,
                             collate_fn=mt_datasets.mt_collate,
                             num_workers=0)

    model = torch.load("./" + context["log_directory"] + "/best_model.pt")

    if cuda_available:
        model.cuda()
    model.eval()

    # create output folder for 3D prediction masks
    path_3Dpred = os.path.join(context['log_directory'], 'pred_masks')
    if not os.path.isdir(path_3Dpred):
        os.makedirs(path_3Dpred)

    metric_fns = [dice_score,  # from ivadomed/utils.py
                  mt_metrics.hausdorff_score,
                  mt_metrics.precision_score,
                  mt_metrics.recall_score,
                  mt_metrics.specificity_score,
                  mt_metrics.intersection_over_union,
                  mt_metrics.accuracy_score]

    metric_mgr = IvadoMetricManager(metric_fns)

    pred_tmp_lst, z_tmp_lst, fname_tmp = [], [], ''
    for i, batch in enumerate(test_loader):
        input_samples, gt_samples = batch["input"], batch["gt"]

        with torch.no_grad():
            if cuda_available:
                test_input = input_samples.cuda()
                test_gt = gt_samples.cuda(non_blocking=True)
            else:
                test_input = input_samples
                test_gt = gt_samples

            if film_bool:
                sample_metadata = batch["input_metadata"]
                test_contrast = [sample_metadata[k]['contrast'] for k in range(len(sample_metadata))]

                test_metadata = [one_hot_encoder.transform([sample_metadata[k]["film_input"]]).tolist()[0] for k in
                                 range(len(sample_metadata))]
                preds = model(test_input, test_metadata)  # Input the metadata related to the input samples
            else:
                preds = model(test_input)

        # WARNING: sample['gt'] is actually the pred in the return sample
        # implementation justification: the other option: rdict['pred'] = preds would require to largely modify mt_transforms
        rdict = {}
        rdict['gt'] = preds.cpu()
        batch.update(rdict)

        # reconstruct 3D image
        for smp_idx in range(len(batch['gt'])):
            # undo transformations
            rdict = {}
            for k in batch.keys():
                rdict[k] = batch[k][smp_idx]
            rdict_undo = val_undo_transform(rdict)

            fname_ref = rdict_undo['input_metadata']['gt_filename']
            if pred_tmp_lst and (fname_ref != fname_tmp or (i == len(test_loader)-1 and smp_idx == len(batch['gt'])-1)):  # new processed file
                # save the completely processed file as a nii
                fname_pred = path_3Dpred + fname_tmp.split('/')[-1]
                fname_pred = fname_pred.split('manual.nii.gz')[0] + 'pred.nii.gz'
                save_nii(pred_tmp_lst, z_tmp_lst, fname_tmp, fname_pred, SLICE_AXIS)
                # re-init pred_stack_lst
                pred_stack_lst, z_tmp_lst = [], []

        # Metrics computation
        gt_npy = gt_samples.numpy().astype(np.uint8)
        gt_npy = gt_npy.squeeze(axis=1)

        preds_npy = preds.data.cpu().numpy()
        preds_npy = threshold_predictions(preds_npy)
        preds_npy = preds_npy.astype(np.uint8)
        preds_npy = preds_npy.squeeze(axis=1)

        metric_mgr(preds_npy, gt_npy)

    metrics_dict = metric_mgr.get_results()
    metric_mgr.reset()
    print(metrics_dict)


def run_main():
    if len(sys.argv) <= 1:
        print("\nivadomed [config.json]\n")
        return

    with open(sys.argv[1], "r") as fhandle:
        context = json.load(fhandle)

    command = context["command"]

    if command == 'train':
        cmd_train(context)
        shutil.copyfile(sys.argv[1], "./" + context["log_directory"] + "/config_file.json")
    elif command == 'test':
        cmd_test(context)


if __name__ == "__main__":
    run_main()

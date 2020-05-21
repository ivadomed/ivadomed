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

from ivadomed import losses as imed_losses
from ivadomed import metrics as imed_metrics
from ivadomed import models as imed_models
from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader, film as imed_film

cudnn.benchmark = True


def cmd_train(context):
    """Main command to train the network.

    :param context: this is a dictionary with all data from the
                    configuration file
    """
    ##### DEFINE DEVICE #####
    device = torch.device("cuda:" + str(context['gpu']) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("Cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_number = int(context["gpu"])
        torch.cuda.set_device(gpu_number)
        print("Using GPU number {}".format(gpu_number))

    # Boolean which determines if the selected architecture is FiLMed-Unet or Unet or Mixup-Unet
    metadata_bool = False if context["metadata"] == "without" else True
    film_bool = (bool(sum(context["film_layers"])) and metadata_bool)

    unet_3D = context["unet_3D"]
    HeMIS = context['HeMIS']
    attention = context["attention_unet"]
    if film_bool:
        context["multichannel"] = False
        HeMIS = False
    elif context["multichannel"]:
        HeMIS = False
    if HeMIS:
        # Initializing the probability of missing modalities for HeMIS.
        # Higher probability means a more missing modalities
        p = context["missing_probability"]

    if bool(sum(context["film_layers"])) and not (metadata_bool):
        print('\tWarning FiLM disabled since metadata is disabled')
    else:

        print('\nArchitecture: {} with a depth of {}.\n' \
              .format('FiLMedUnet' if film_bool else 'HeMIS-Unet' if HeMIS else "Attention UNet" if attention else
        '3D Unet' if unet_3D else "Unet", context['depth']))

    mixup_bool = False if film_bool else bool(context["mixup_bool"])
    mixup_alpha = float(context["mixup_alpha"])

    if not film_bool and mixup_bool:
        print('\twith Mixup (alpha={})\n'.format(mixup_alpha))
    if context["metadata"] == "mri_params":
        print('\tInclude subjects with acquisition metadata available only.\n')
    else:
        print('\tInclude all subjects, with or without acquisition metadata.\n')
    if context['multichannel']:
        print('\tUsing multichannel model with modalities {}.\n'.format(context['contrast_train_validation']))

    # Write the metrics, images, etc to TensorBoard format
    writer = SummaryWriter(log_dir=context["log_directory"])

    # Compose training transforms
    train_transform = imed_transforms.Compose(context["transformation_training"])

    # Compose validation transforms
    val_transform = imed_transforms.Compose(context["transformation_validation"])

    # Randomly split dataset between training / validation / testing
    if context.get("split_path") is None:
        train_lst, valid_lst, test_lst = imed_loader_utils.split_dataset(path_folder=context["bids_path"],
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

    # This code will iterate over the folders and load the data, filtering
    # the slices without labels and then concatenating all the datasets together
    ds_train = imed_loader.load_dataset(train_lst, train_transform, context)

    # if ROICrop2D in transform, then apply SliceFilter to ROI slices
    # todo: not supported by the adaptative loader
    if 'ROICrop2D' in context["transformation_training"].keys():
        ds_train = imed_loader_utils.filter_roi(ds_train, nb_nonzero_thr=context["slice_filter_roi"])

    if film_bool:  # normalize metadata before sending to the network
        if context["metadata"] == "mri_params":
            metadata_vector = ["RepetitionTime", "EchoTime", "FlipAngle"]
            metadata_clustering_models = imed_film.clustering_fit(ds_train.metadata, metadata_vector)
        else:
            metadata_clustering_models = None

        ds_train, train_onehotencoder = imed_film.normalize_metadata(ds_train,
                                                                     metadata_clustering_models,
                                                                     context["debugging"],
                                                                     context["metadata"],
                                                                     True)

    if not unet_3D:
        print(f"Loaded {len(ds_train)} {context['slice_axis']} slices for the training set.")
    else:
        print(f"Loaded {len(ds_train)} volumes of size {context['length_3D']} for the training set.")

    if context['balance_samples'] and not HeMIS:
        sampler_train = imed_loader_utils.BalancedSampler(ds_train)
        shuffle_train = False
    else:
        sampler_train, shuffle_train = None, True

    train_loader = DataLoader(ds_train, batch_size=context["batch_size"],
                              shuffle=shuffle_train, pin_memory=True, sampler=sampler_train,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=0)
    print("Validation")

    # Validation dataset ------------------------------------------------------
    ds_val = imed_loader.load_dataset(valid_lst, val_transform, context)

    # if ROICrop2D in transform, then apply SliceFilter to ROI slices
    if 'ROICrop2D' in context["transformation_validation"].keys():
        ds_val = imed_loader_utils.filter_roi(ds_val, nb_nonzero_thr=context["slice_filter_roi"])

    if film_bool:  # normalize metadata before sending to network
        ds_val = imed_film.normalize_metadata(ds_val,
                                              metadata_clustering_models,
                                              context["debugging"],
                                              context["metadata"],
                                              False)

    if not unet_3D:
        print(f"Loaded {len(ds_val)} {context['slice_axis']} slices for the validation set.")
    else:
        print(
            f"Loaded {len(ds_val)} volumes of size {context['length_3D']} for the validation set.")

    if context['balance_samples'] and not HeMIS:
        sampler_val = imed_loader_utils.BalancedSampler(ds_val)
        shuffle_val = False
    else:
        sampler_val, shuffle_val = None, True

    val_loader = DataLoader(ds_val, batch_size=context["batch_size"],
                            shuffle=shuffle_val, pin_memory=True, sampler=sampler_val,
                            collate_fn=imed_loader_utils.imed_collate,
                            num_workers=0)
    if film_bool:
        n_metadata = len([ll for l in train_onehotencoder.categories_ for ll in l])
    else:
        n_metadata = None

    # Traditional U-Net model
    if context['multichannel']:
        in_channel = len(context['contrast_train_validation'])
    else:
        in_channel = 1

    if len(context['target_suffix']) > 1:
        # + 1 for background class
        out_channel = len(context["target_suffix"]) + 1
    else:
        out_channel = 1

    if context['retrain_model'] is None:
        if HeMIS:
            model = imed_models.HeMISUnet(modalities=context['contrast_train_validation'],
                                          out_channel=out_channel,
                                          depth=context['depth'],
                                          drop_rate=context["dropout_rate"],
                                          bn_momentum=context["batch_norm_momentum"])
        elif unet_3D:
            model = imed_models.UNet3D(in_channels=in_channel,
                                       n_classes=out_channel,
                                       drop_rate=context["dropout_rate"],
                                       momentum=context["batch_norm_momentum"],
                                       base_n_filter=context["n_filters"],
                                       attention=attention)
        else:
            model = imed_models.Unet(in_channel=in_channel,
                                     out_channel=out_channel,
                                     depth=context['depth'],
                                     film_layers=context["film_layers"],
                                     n_metadata=n_metadata,
                                     drop_rate=context["dropout_rate"],
                                     bn_momentum=context["batch_norm_momentum"],
                                     film_bool=film_bool)
    else:
        # Load pretrained model
        model = torch.load(context['retrain_model'])

        # Freeze first layers and reset last layers
        model = imed_models.set_model_for_retrain(model,
                                                  retrain_fraction=context['retrain_fraction'])

    if cuda_available:
        model.cuda()

    num_epochs = context["num_epochs"]
    initial_lr = context["initial_lr"]

    # Using Adam
    step_scheduler_batch = False
    # filter out the parameters you are going to fine-tuning
    params_to_opt = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_opt, lr=initial_lr)
    if context["lr_scheduler"]["name"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif context["lr_scheduler"]["name"] == "CosineAnnealingWarmRestarts":
        T_0 = context["lr_scheduler"]["T_0"]
        T_mult = context["lr_scheduler"]["T_mult"]
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    elif context["lr_scheduler"]["name"] == "CyclicLR":
        base_lr, max_lr = context["lr_scheduler"]["base_lr"], context["lr_scheduler"]["max_lr"]
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr, max_lr, mode="triangular2", cycle_momentum=False)
        step_scheduler_batch = True
    else:
        print(
            "Unknown LR Scheduler name, please choose between 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', or 'CyclicLR'")
        exit()

    # Create dict containing gammas and betas after each FiLM layer.
    depth = context["depth"]
    gammas_dict = {i: [] for i in range(1, 2 * depth + 3)}
    betas_dict = {i: [] for i in range(1, 2 * depth + 3)}

    # Create a list containing the contrast of all batch images
    var_contrast_list = []

    # Loss
    if context["loss"]["name"] in ["dice", "cross_entropy", "focal", "gdl", "focal_dice", "multi_class_dice"]:
        if context["loss"]["name"] == "cross_entropy":
            loss_fct = nn.BCELoss()

        elif context["loss"]["name"] == "focal":
            loss_fct = imed_losses.FocalLoss(gamma=context["loss"]["params"]["gamma"],
                                             alpha=context["loss"]["params"]["alpha"])
            print("\nLoss function: {}, with gamma={}, alpha={}.\n".format(context["loss"]["name"],
                                                                           context["loss"]["params"]["gamma"],
                                                                           context["loss"]["params"]["alpha"]))
        elif context["loss"]["name"] == "gdl":
            loss_fct = imed_losses.GeneralizedDiceLoss()

        elif context["loss"]["name"] == "focal_dice":
            loss_fct = imed_losses.FocalDiceLoss(beta=context["loss"]["params"]["beta"],
                                                 gamma=context["loss"]["params"]["gamma"],
                                                 alpha=context["loss"]["params"]["alpha"])
            print("\nLoss function: {}, with beta={}, gamma={} and alpha={}.\n".format(context["loss"]["name"],
                                                                                       context["loss"]["params"][
                                                                                           "beta"],
                                                                                       context["loss"]["params"][
                                                                                           "gamma"],
                                                                                       context["loss"]["params"][
                                                                                           "alpha"]))
        elif context["loss"]["name"] == "multi_class_dice":
            loss_fct = imed_losses.MultiClassDiceLoss(classes_of_interest=
                                                      context["loss"]["params"]["classes_of_interest"])

        if not context["loss"]["name"].startswith("focal"):
            print("\nLoss function: {}.\n".format(context["loss"]["name"]))

    else:
        print("Unknown Loss function, please choose between 'dice', 'focal', 'focal_dice', 'gdl', 'cross_entropy' "
              "or 'multi_class_dice'")
        exit()

    # Training loop -----------------------------------------------------------

    best_training_dice, best_training_loss, best_validation_loss, best_validation_dice = float("inf"), float(
        "inf"), float("inf"), float("inf")

    patience = context["early_stopping_patience"]
    patience_count = 0
    epsilon = context["early_stopping_epsilon"]
    val_losses = []

    metric_fns = [imed_metrics.dice_score,
                  imed_metrics.multi_class_dice_score,
                  imed_metrics.hausdorff_3D_score,
                  imed_metrics.precision_score,
                  imed_metrics.recall_score,
                  imed_metrics.specificity_score,
                  imed_metrics.intersection_over_union,
                  imed_metrics.accuracy_score]

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        start_time = time.time()

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)

        model.train()
        train_loss_total, dice_train_loss_total = 0.0, 0.0

        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples = batch["input"] if not HeMIS \
                                            else imed_utils.unstack_tensors(batch["input"]), batch["gt"]

            # mixup data
            if mixup_bool and not film_bool:
                input_samples, gt_samples, lambda_tensor = imed_utils.mixup(
                    input_samples, gt_samples, mixup_alpha)

                # if debugging and first epoch, then save samples as png in log folder
                if context["debugging"] and epoch == 1 and random.random() < 0.1:
                    mixup_folder = os.path.join(context["log_directory"], 'mixup')
                    if not os.path.isdir(mixup_folder):
                        os.makedirs(mixup_folder)
                    random_idx = np.random.randint(0, input_samples.size()[0])
                    val_gt = np.unique(gt_samples.data.numpy()[random_idx, 0, :, :])
                    mixup_fname_pref = os.path.join(mixup_folder, str(i).zfill(3) + '_' + str(
                        lambda_tensor.data.numpy()[0]) + '_' + str(random_idx).zfill(3) + '.png')
                    imed_utils.save_mixup_sample(input_samples.data.numpy()[random_idx, 0, :, :],
                                                 gt_samples.data.numpy()[random_idx, 0, :, :],
                                                 mixup_fname_pref)

            # The variable sample_metadata is where the MRI physics parameters are

            if cuda_available:
                var_input = imed_utils.cuda(input_samples)
                var_gt = imed_utils.cuda(gt_samples, non_blocking=True)
            else:
                var_input = input_samples
                var_gt = gt_samples

            if film_bool:
                # var_contrast is the list of the batch sample's contrasts (eg T2w, T1w).
                sample_metadata = batch["input_metadata"]
                var_contrast = [sample_metadata[0][k]['contrast'] for k in range(len(sample_metadata[0]))]
                var_metadata = [train_onehotencoder.transform([sample_metadata[0][k]['film_input']]).tolist()[0]
                                for k in range(len(sample_metadata[0]))]

                # Input the metadata related to the input samples
                preds = model(var_input, var_metadata)
            elif HeMIS:
                missing_mod = batch["Missing_mod"]
                preds = model(var_input, missing_mod)
            else:
                preds = model(var_input)

            if context["loss"]["name"] == "dice":
                loss = - imed_losses.dice_loss(preds, var_gt)

            else:
                loss = loss_fct(preds, var_gt)
                dice_train_loss_total += imed_losses.dice_loss(preds, var_gt).item()
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if step_scheduler_batch:
                scheduler.step()

            num_steps += 1

            if i == 0:
                imed_utils.save_tensorboard_img(writer, epoch, "Train", input_samples, gt_samples, preds, unet_3D)

        train_loss_total_avg = train_loss_total / num_steps
        if not step_scheduler_batch:
            scheduler.step()

        tqdm.write(f"Epoch {epoch} training loss: {train_loss_total_avg:.4f}.")
        if context["loss"]["name"] != 'dice':
            dice_train_loss_total_avg = dice_train_loss_total / num_steps
            tqdm.write(f"\tDice training loss: {dice_train_loss_total_avg:.4f}.")

        # In case of curriculum Learning we need to update the loader
        if HeMIS:
            # Increase the probability of a missing modality
            p = p ** (context["missing_probability_growth"])
            ds_train.update(p=p)
            train_loader = DataLoader(ds_train, batch_size=context["batch_size"],
                                      shuffle=shuffle_train, pin_memory=True, sampler=sampler_train,
                                      collate_fn=imed_loader_utils.imed_collate,
                                      num_workers=0)

        # Validation loop -----------------------------------------------------
        model.eval()
        val_loss_total, dice_val_loss_total = 0.0, 0.0
        num_steps = 0

        metric_mgr = imed_metrics.MetricManager(metric_fns)

        for i, batch in enumerate(val_loader):
            input_samples, gt_samples = batch["input"] if not HeMIS \
                                            else imed_utils.unstack_tensors(batch["input"]), batch["gt"]

            with torch.no_grad():
                if cuda_available:
                    var_input = imed_utils.cuda(input_samples)
                    var_gt = imed_utils.cuda(gt_samples, non_blocking=True)
                else:
                    var_input = input_samples
                    var_gt = gt_samples

                if film_bool:
                    sample_metadata = batch["input_metadata"]
                    # var_contrast is the list of the batch sample's contrasts (eg T2w, T1w).
                    var_contrast = [sample_metadata[0][k]['contrast']
                                    for k in range(len(sample_metadata[0]))]

                    var_metadata = [train_onehotencoder.transform([sample_metadata[0][k]['film_input']]).tolist()[0]
                                    for k in range(len(sample_metadata[0]))]

                    # Input the metadata related to the input samples
                    preds = model(var_input, var_metadata)
                elif HeMIS:
                    missing_mod = batch["Missing_mod"]
                    preds = model(var_input, missing_mod)

                else:
                    preds = model(var_input)

                if context["loss"]["name"] == "dice":
                    loss = - imed_losses.dice_loss(preds, var_gt)

                else:
                    loss = loss_fct(preds, var_gt)
                    dice_val_loss_total += imed_losses.dice_loss(preds, var_gt).item()
                val_loss_total += loss.item()

            # Metrics computation
            gt_npy = gt_samples.numpy().astype(np.uint8)

            preds_npy = preds.data.cpu().numpy()
            if context["binarize_prediction"]:
                preds_npy = imed_postpro.threshold_predictions(preds_npy)
            preds_npy = preds_npy.astype(np.uint8)

            metric_mgr(preds_npy, gt_npy)

            num_steps += 1

            # Only write sample at the first step
            if i == 0:
                imed_utils.save_tensorboard_img(writer, epoch, "Validation", input_samples, gt_samples, preds, unet_3D)

            # Store the values of gammas and betas after the last epoch for each batch
            if film_bool and epoch == num_epochs and i < int(len(ds_val) / context["batch_size"]) + 1:

                # Get all the contrasts of all batches
                var_contrast_list.append(var_contrast)

                # Get the list containing the number of film layers
                film_layers = context["film_layers"]

                # Fill the lists of gammas and betas
                for idx in [i for i, x in enumerate(film_layers) if x]:
                    if idx < depth:
                        layer_cur = model.encoder.down_path[idx * 3 + 1]
                    elif idx == depth:
                        layer_cur = model.encoder.film_bottom
                    elif idx == depth * 2 + 1:
                        layer_cur = model.decoder.last_film
                    else:
                        layer_cur = model.decoder.up_path[(idx - depth - 1) * 2 + 1]

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
        joblib.dump(metadata_clustering_models, "./" +
                    context["log_directory"] + "/clustering_models.joblib")
        joblib.dump(train_onehotencoder, "./" +
                    context["log_directory"] + "/one_hot_encoder.joblib")

        # Convert list of gammas/betas into numpy arrays
        gammas_dict = {i: np.array(gammas_dict[i]) for i in range(1, 2 * depth + 3)}
        betas_dict = {i: np.array(betas_dict[i]) for i in range(1, 2 * depth + 3)}

        # Save the numpy arrays for gammas/betas inside files.npy in log_directory
        for i in range(1, 2 * depth + 3):
            np.save(context["log_directory"] + f"/gamma_layer_{i}.npy", gammas_dict[i])
            np.save(context["log_directory"] + f"/beta_layer_{i}.npy", betas_dict[i])

        # Convert into numpy and save the contrasts of all batch images
        contrast_images = np.array(var_contrast_list)
        np.save(context["log_directory"] + "/contrast_images.npy", contrast_images)

    writer.close()
    return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss


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
    elif command == 'eval':
        cmd_eval(context)


if __name__ == "__main__":
    run_main()

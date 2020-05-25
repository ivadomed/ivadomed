import os
import random
import time

import joblib
import numpy as np

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

def train(model_params, dataset_train, dataset_val, log_directory, cuda_available=True, balance_samples=False,
          mixup_params=None):
    """Main command to train the network.

    Args:
        model_params (dict): Model's parameters.
        dataset_train (imed_loader): Training dataset
        dataset_val (imed_loader): Validation dataset
        log_directory (string):
        cuda_available (Bool):
        balance_samples (Bool):
        mixup_params (float): alpha parameter
    Returns:
        XX
    """
    # Write the metrics, images, etc to TensorBoard format
    writer = SummaryWriter(log_dir=log_directory)

    if context['balance_samples'] and not HeMIS:
        sampler_train = imed_loader_utils.BalancedSampler(ds_train)
        shuffle_train = False
    else:
        sampler_train, shuffle_train = None, True

    train_loader = DataLoader(ds_train, batch_size=context["batch_size"],
                              shuffle=shuffle_train, pin_memory=True, sampler=sampler_train,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=0)

    # BALANCE SAMPLES
    conditions = [balance_samples, model_params["name"] != "HeMIS"]
    sampler_val, suffler_val = get_sampler(balance_bool=all(conditions))
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
                    mixup_folder = os.path.join(log_directory, 'mixup')
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
            torch.save(model, "./" + log_directory + "/best_model.pt")

        # Early stopping : break if val loss doesn't improve by at least epsilon percent for N=patience epochs
        val_losses.append(val_loss_total_avg)

        if epoch > 1:
            if (val_losses[-2] - val_losses[-1]) * 100 / abs(val_losses[-1]) < epsilon:
                patience_count += 1
        if patience_count >= patience:
            print(f"Stopping training due to {patience} epochs without improvements")
            break

    # Save final model
    torch.save(model, "./" + log_directory + "/final_model.pt")
    if film_bool:  # save clustering and OneHotEncoding models
        joblib.dump(metadata_clustering_models, "./" +
                    log_directory + "/clustering_models.joblib")
        joblib.dump(train_onehotencoder, "./" +
                    log_directory + "/one_hot_encoder.joblib")

        # Convert list of gammas/betas into numpy arrays
        gammas_dict = {i: np.array(gammas_dict[i]) for i in range(1, 2 * depth + 3)}
        betas_dict = {i: np.array(betas_dict[i]) for i in range(1, 2 * depth + 3)}

        # Save the numpy arrays for gammas/betas inside files.npy in log_directory
        for i in range(1, 2 * depth + 3):
            np.save(log_directory + f"/gamma_layer_{i}.npy", gammas_dict[i])
            np.save(log_directory + f"/beta_layer_{i}.npy", betas_dict[i])

        # Convert into numpy and save the contrasts of all batch images
        contrast_images = np.array(var_contrast_list)
        np.save(log_directory + "/contrast_images.npy", contrast_images)

    writer.close()
    return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss


def get_sampler(ds, balance_bool):
    """Get sampler.

    Args:
        ds (BidsDataset):
        balance_bool (Bool):
    Returns:
        Sampler, Bool: Sampler and boolean for shuffling
    """
    if balance_bool:
        return imed_loader_utils.BalancedSampler(ds), False
    else:
        return None, True

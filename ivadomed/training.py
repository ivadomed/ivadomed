import copy
import os
import time

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
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils
import datetime

cudnn.benchmark = True


def train(model_params, dataset_train, dataset_val, training_params, log_directory, device,
          cuda_available=True, metric_fns=None, debugging=False):
    """Main command to train the network.

    Args:
        model_params (dict): Model's parameters.
        dataset_train (imed_loader): Training dataset.
        dataset_val (imed_loader): Validation dataset.
        training_params (dict):
        log_directory (str): Folder where log files, best and final models are saved.
        device (str): Indicates the CPU or GPU ID.
        cuda_available (bool): If True, CUDA is available.
        metric_fns (list): List of metrics, see :mod:`ivadomed.metrics`.
        debugging (bool): If True, extended verbosity and intermediate outputs.

    Returns:
        float, float, float, float: best_training_dice, best_training_loss, best_validation_dice, best_validation_loss.
    """
    # Write the metrics, images, etc to TensorBoard format
    writer = SummaryWriter(log_dir=log_directory)

    # BALANCE SAMPLES AND PYTORCH LOADER
    conditions = all([training_params["balance_samples"], model_params["name"] != "HeMIS"])
    sampler_train, shuffle_train = get_sampler(dataset_train, conditions)

    train_loader = DataLoader(dataset_train, batch_size=training_params["batch_size"],
                              shuffle=shuffle_train, pin_memory=True, sampler=sampler_train,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=0)

    if dataset_val:
        sampler_val, shuffle_val = get_sampler(dataset_val, conditions)

        val_loader = DataLoader(dataset_val, batch_size=training_params["batch_size"],
                                shuffle=shuffle_val, pin_memory=True, sampler=sampler_val,
                                collate_fn=imed_loader_utils.imed_collate,
                                num_workers=0)

    # GET MODEL
    if training_params["transfer_learning"]["retrain_model"]:
        print("\nLoading pretrained model's weights: {}.")
        print("\tFreezing the {}% first layers.".format(
            100 - training_params["transfer_learning"]['retrain_fraction'] * 100.))
        old_model_path = training_params["transfer_learning"]["retrain_model"]
        fraction = training_params["transfer_learning"]['retrain_fraction']
        # Freeze first layers and reset last layers
        model = imed_models.set_model_for_retrain(old_model_path, retrain_fraction=fraction, map_location=device)
    else:
        print("\nInitialising model's weights from scratch.")
        model_class = getattr(imed_models, model_params["name"])
        model = model_class(**model_params)
    if cuda_available:
        model.cuda()

    num_epochs = training_params["training_time"]["num_epochs"]

    # OPTIMIZER
    initial_lr = training_params["scheduler"]["initial_lr"]
    # filter out the parameters you are going to fine-tuning
    params_to_opt = filter(lambda p: p.requires_grad, model.parameters())
    # Using Adam
    optimizer = optim.Adam(params_to_opt, lr=initial_lr)
    scheduler, step_scheduler_batch = get_scheduler(training_params["scheduler"]["lr_scheduler"], optimizer, num_epochs)
    print("\nScheduler parameters: {}".format(training_params["scheduler"]["lr_scheduler"]))

    # Create dict containing gammas and betas after each FiLM layer.
    if model_params["name"] == "FiLMedUnet":
        gammas_dict = {i: [] for i in range(1, 2 * model_params["depth"] + 3)}
        betas_dict = {i: [] for i in range(1, 2 * model_params["depth"] + 3)}
        contrast_list = []

    # LOSS
    print("\nSelected Loss: {}".format(training_params["loss"]["name"]))
    print("\twith the parameters: {}".format(
        [training_params["loss"][k] for k in training_params["loss"] if k != "name"]))
    loss_fct = get_loss_function(copy.copy(training_params["loss"]))
    loss_dice_fct = imed_losses.DiceLoss()  # For comparison when another loss is used

    # INIT TRAINING VARIABLES
    best_training_dice, best_training_loss = float("inf"), float("inf")
    best_validation_loss, best_validation_dice = float("inf"), float("inf")
    patience_count = 0
    begin_time = time.time()

    # EPOCH LOOP
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        start_time = time.time()

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)

        # Training loop -----------------------------------------------------------
        model.train()
        train_loss_total, train_dice_loss_total = 0.0, 0.0
        num_steps = 0
        for i, batch in enumerate(train_loader):
            # GET SAMPLES
            if model_params["name"] == "HeMISUnet":
                input_samples = imed_utils.cuda(imed_utils.unstack_tensors(batch["input"]), cuda_available)
            else:
                input_samples = imed_utils.cuda(batch["input"], cuda_available)
            gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)

            # MIXUP
            if training_params["mixup_alpha"]:
                input_samples, gt_samples = imed_utils.mixup(input_samples, gt_samples, training_params["mixup_alpha"],
                                                             debugging and epoch == 1, log_directory)

            # RUN MODEL
            if model_params["name"] in ["HeMISUnet", "FiLMedUnet"]:
                metadata = get_metadata(batch["input_metadata"], model_params)
                preds = model(input_samples, metadata)
            else:
                preds = model(input_samples)

            # LOSS
            loss = loss_fct(preds, gt_samples)
            train_loss_total += loss.item()
            train_dice_loss_total += loss_dice_fct(preds, gt_samples).item()

            # UPDATE OPTIMIZER
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step_scheduler_batch:
                scheduler.step()
            num_steps += 1

            if i == 0 and debugging:
                imed_utils.save_tensorboard_img(writer, epoch, "Train", input_samples, gt_samples, preds,
                                                is_three_dim=model_params["name"].endswith("3D"))

        if not step_scheduler_batch:
            scheduler.step()

        # TRAINING LOSS
        train_loss_total_avg = train_loss_total / num_steps
        msg = "Epoch {} training loss: {:.4f}.".format(epoch, train_loss_total_avg)
        train_dice_loss_total_avg = train_dice_loss_total / num_steps
        if training_params["loss"]["name"] != "DiceLoss":
            msg += "\tDice training loss: {:.4f}.".format(train_dice_loss_total_avg)
        tqdm.write(msg)

        # CURRICULUM LEARNING
        if model_params["name"] == "HeMISUnet":
            # Increase the probability of a missing modality
            model_params["missing_probability"] **= model_params["missing_probability_growth"]
            dataset_train.update(p=model_params["missing_probability"])

        # Validation loop -----------------------------------------------------
        model.eval()
        val_loss_total, val_dice_loss_total = 0.0, 0.0
        num_steps = 0
        metric_mgr = imed_metrics.MetricManager(metric_fns)
        if dataset_val:
            for i, batch in enumerate(val_loader):
                with torch.no_grad():
                    # GET SAMPLES
                    if model_params["name"] == "HeMISUnet":
                        input_samples = imed_utils.cuda(imed_utils.unstack_tensors(batch["input"]), cuda_available)
                    else:
                        input_samples = imed_utils.cuda(batch["input"], cuda_available)
                    gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)

                    # RUN MODEL
                    if model_params["name"] in ["HeMISUnet", "FiLMedUnet"]:
                        metadata = get_metadata(batch["input_metadata"], model_params)
                        preds = model(input_samples, metadata)
                    else:
                        preds = model(input_samples)

                    # LOSS
                    loss = loss_fct(preds, gt_samples)
                    val_loss_total += loss.item()
                    val_dice_loss_total += loss_dice_fct(preds, gt_samples).item()

                num_steps += 1

                # METRICS COMPUTATION
                gt_npy = gt_samples.cpu().numpy().astype(np.uint8)
                preds_npy = preds.data.cpu().numpy()
                metric_mgr(preds_npy.astype(np.uint8), gt_npy)

                if i == 0 and debugging:
                    imed_utils.save_tensorboard_img(writer, epoch, "Validation", input_samples, gt_samples, preds,
                                                    is_three_dim=model_params["name"].endswith("3D"))

                if model_params["name"] == "FiLMedUnet" and debugging and epoch == num_epochs \
                        and i < int(len(dataset_val) / training_params["batch_size"]) + 1:
                    # Store the values of gammas and betas after the last epoch for each batch
                    gammas_dict, betas_dict, contrast_list = store_film_params(gammas_dict, betas_dict, contrast_list,
                                                                               batch['input_metadata'], model,
                                                                               model_params["film_layers"],
                                                                               model_params["depth"])

            # METRICS COMPUTATION FOR CURRENT EPOCH
            val_loss_total_avg_old = val_loss_total_avg if epoch > 1 else None
            metrics_dict = metric_mgr.get_results()
            metric_mgr.reset()
            writer.add_scalars('Validation/Metrics', metrics_dict, epoch)
            val_loss_total_avg = val_loss_total / num_steps
            writer.add_scalars('losses', {
                'train_loss': train_loss_total_avg,
                'val_loss': val_loss_total_avg,
            }, epoch)
            msg = "Epoch {} validation loss: {:.4f}.".format(epoch, val_loss_total_avg)
            val_dice_loss_total_avg = val_dice_loss_total / num_steps
            if training_params["loss"]["name"] != "DiceLoss":
                msg += "\tDice validation loss: {:.4f}.".format(val_dice_loss_total_avg)
            tqdm.write(msg)
            end_time = time.time()
            total_time = end_time - start_time
            tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

            # UPDATE BEST RESULTS
            if val_loss_total_avg < best_validation_loss:
                best_validation_loss, best_training_loss = val_loss_total_avg, train_loss_total_avg
                best_validation_dice, best_training_dice = val_dice_loss_total_avg, train_dice_loss_total_avg
                model_path = os.path.join(log_directory, "best_model.pt")
                torch.save(model, model_path)

            # EARLY STOPPING
            if epoch > 1:
                val_diff = (val_loss_total_avg_old - val_loss_total_avg) * 100 / abs(val_loss_total_avg)
                if val_diff < training_params["training_time"]["early_stopping_epsilon"]:
                    patience_count += 1
                if patience_count >= training_params["training_time"]["early_stopping_patience"]:
                    print("Stopping training due to {} epochs without improvements".format(patience_count))
                    break

    # Save final model
    final_model_path = os.path.join(log_directory, "final_model.pt")
    torch.save(model, final_model_path)
    if model_params["name"] == "FiLMedUnet" and debugging:
        save_film_params(gammas_dict, betas_dict, contrast_list, model_params["depth"], log_directory)

    # Convert best model to ONNX and save it in model directory
    best_model_path = os.path.join(log_directory, model_params["folder_name"], model_params["folder_name"] + ".onnx")
    imed_utils.save_onnx_model(torch.load(model_path), input_samples, best_model_path)

    writer.close()
    final_time = time.time()
    duration_time = final_time - begin_time
    print('begin ' + time.strftime('%H:%M:%S', time.localtime(begin_time)) + "| End " +
          time.strftime('%H:%M:%S', time.localtime(final_time)) +
          "| duration " + str(datetime.timedelta(seconds=duration_time)))

    return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss


def get_sampler(ds, balance_bool):
    """Get sampler.

    Args:
        ds (BidsDataset): BidsDataset object.
        balance_bool (bool): If True, a sampler is generated that balance positive and negative samples.

    Returns:
        If balance_bool is True: Returns BalancedSampler, Bool: Sampler and boolean for shuffling (set to False).
        Otherwise: Returns None and True.
    """
    if balance_bool:
        return imed_loader_utils.BalancedSampler(ds), False
    else:
        return None, True


def get_scheduler(params, optimizer, num_epochs=0):
    """Get scheduler.

    Args:
        params (dict): scheduler parameters, see `PyTorch documentation <https://pytorch.org/docs/stable/optim.html>`__
        optimizer (torch optim):
        num_epochs (int): number of epochs.

    Returns:
        torch.optim, bool, which indicates if the scheduler is updated for each batch (True), or for each epoch (False).
    """
    step_scheduler_batch = False
    scheduler_name = params["name"]
    del params["name"]
    if scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
    elif scheduler_name == "CyclicLR":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, **params, mode="triangular2", cycle_momentum=False)
        step_scheduler_batch = True
    else:
        print(
            "Unknown LR Scheduler name, please choose between 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts',"
            "or 'CyclicLR'")
        exit()
    return scheduler, step_scheduler_batch


def get_loss_function(params):
    """Get Loss function.

    Args:
        params (dict): See :mod:`ivadomed.losses`.

    Returns:
        imed_losses object.
    """
    # Loss function name
    loss_name = params["name"]
    del params["name"]

    # Check if implemented
    loss_function_available = ["DiceLoss", "FocalLoss", "GeneralizedDiceLoss", "FocalDiceLoss", "MultiClassDiceLoss",
                               "BinaryCrossEntropyLoss", "TverskyLoss", "FocalTverskyLoss", "AdapWingLoss", "L2loss",
                               "LossCombination"]
    if loss_name not in loss_function_available:
        print("Unknown Loss function, please choose between {}".format(loss_function_available))
        exit()

    loss_class = getattr(imed_losses, loss_name)
    loss_fct = loss_class(**params)
    return loss_fct


def get_metadata(metadata, model_params):
    """Get metadata during batch loop.

    Args:
        metadata (batch):
        model_params (dict):

    Returns:
        If FiLMedUnet, Returns a list of metadata, that have been transformed by the One Hot Encoder.
        If HeMISUnet, Returns a numpy array where each row represents a sample and each column represents a contrast.
    """
    if model_params["name"] == "HeMISUnet":
        return np.array([m[0]["missing_mod"] for m in metadata])
    else:
        return [model_params["film_onehotencoder"].transform([metadata[0][k]['film_input']]).tolist()[0]
                for k in range(len(metadata[0]))]


def store_film_params(gammas, betas, contrasts, metadata, model, film_layers, depth):
    """Store FiLM params.

    Args:
        gammas (dict):
        betas (dict):
        contrasts (list): list of the batch sample's contrasts (eg T2w, T1w)
        metadata (list):
        model (nn.Module):
        film_layers (list):
        depth (int):

    Returns:
        dict, dict: gammas, betas
    """
    new_contrast = [metadata[0][k]['contrast'] for k in range(len(metadata[0]))]
    contrasts.append(new_contrast)
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

        gammas[idx + 1].append(layer_cur.gammas[:, :, 0, 0].cpu().numpy())
        betas[idx + 1].append(layer_cur.betas[:, :, 0, 0].cpu().numpy())
    return gammas, betas, contrasts


def save_film_params(gammas, betas, contrasts, depth, ofolder):
    """Save FiLM params as npy files.

    These parameters can be further used for visualisation purposes. They are saved in the `ofolder` with `.npy` format.

    Args:
        gammas (dict):
        betas (dict):
        contrasts (list): list of the batch sample's contrasts (eg T2w, T1w)
        depth (int):
        ofolder (str):

    """
    # Convert list of gammas/betas into numpy arrays
    gammas_dict = {i: np.array(gammas[i]) for i in range(1, 2 * depth + 3)}
    betas_dict = {i: np.array(betas[i]) for i in range(1, 2 * depth + 3)}

    # Save the numpy arrays for gammas/betas inside files.npy in log_directory
    for i in range(1, 2 * depth + 3):
        gamma_layer_path = os.path.join(ofolder, "gamma_layer_{}.npy".format(i))
        np.save(gamma_layer_path, gammas_dict[i])
        beta_layer_path = os.path.join(ofolder, "beta_layer_{}.npy".format(i))
        np.save(beta_layer_path, betas_dict[i])

    # Convert into numpy and save the contrasts of all batch images
    contrast_images = np.array(contrasts)
    contrast_path = os.path.join(ofolder, "contrast_image.npy")
    np.save(contrast_path, contrast_images)

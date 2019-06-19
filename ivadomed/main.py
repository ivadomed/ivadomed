import sys
import json
import time
import shutil
import pickle

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torchvision.utils as vutils
from torch import optim

from medicaltorch import transforms as mt_transforms
from medicaltorch import datasets as mt_datasets
from medicaltorch import losses as mt_losses
from medicaltorch import filters as mt_filters
from medicaltorch import metrics as mt_metrics

from tensorboardX import SummaryWriter

from tqdm import tqdm

from ivadomed import loader as loader
from ivadomed import models

import numpy as np

from PIL import Image

cudnn.benchmark = True


def threshold_predictions(predictions, thr=0.5):
    """This function will threshold predictions.

    :param predictions: input data (predictions)
    :param thr: threshold to use, default to 0.5
    :return: thresholded input
    """
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


class SliceFilter(mt_filters.SliceFilter):
    """This class extends the SliceFilter that already
    filters for empty labels and inputs. It will filter
    slices that has only zeros after cropping. To avoid
    feeding empty inputs into the network.
    """
    def __call__(self, sample):
        super_ret = super().__call__(sample)

        # Already filtered by base class
        if not super_ret:
            return super_ret

        # Filter slices where there are no values after cropping
        input_img = Image.fromarray(sample['input'], mode='F')
        input_cropped = transforms.functional.center_crop(input_img, (128, 128))
        input_cropped = np.array(input_cropped)
        count = np.count_nonzero(input_cropped)

        if count <= 0:
            return False

        return True


def cmd_train(context):
    """Main command do train the network.

    :param context: this is a dictionary with all data from the
                    configuration file:
                        - 'command': run the specified command (e.g. train, test)
                        - 'gpu': ID of the used GPU
                        - 'bids_path_train': list of relative paths of the BIDS folders of each training center
                        - 'bids_path_validation': list of relative paths of the BIDS folders of each validation center
                        - 'bids_path_test': list of relative paths of the BIDS folders of each test center
                        - 'contrast_train_validation': list of image modalities included in the training and validation datasets
                        - 'contrast_test': list of image modalities included in the testing dataset
                        - 'batch_size'
                        - 'dropout_rate'
                        - 'batch_norm_momentum'
                        - 'num_epochs'
                        - 'initial_lr': initial learning rate
                        - 'log_directory': folder name where log files are saved
                        - 'film': indicates if FiLM is used or not
                        - 'debugging': allows extended verbosity and intermediate outputs
    """
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

    # These are the training transformations
    train_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((128, 128)),
        mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                       sigma_range=(3.5, 4.0),
                                       p=0.3),
        mt_transforms.RandomAffine(degrees=4.6,
                                   scale=(0.98, 1.02),
                                   translate=(0.03, 0.03)),
        mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    # These are the validation/testing transformations
    val_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((128, 128)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    # This code will iterate over the folders and load the data, filtering
    # the slices without labels and then concatenating all the datasets together
    train_datasets, train_metadata = [], []
    for bids_ds in tqdm(context["bids_path_train"], desc="Loading training set"):
        ds_train = loader.BidsDataset(bids_ds,
                                      contrast_lst=context["contrast_train_validation"],
                                      transform=train_transform,
                                      slice_filter_fn=SliceFilter())
        train_datasets.append(ds_train)
        train_metadata.append(ds_train.metadata)  # store the metadata of the training data, used for fitting the clustering models

    if context["film"]:  # normalize metadata before sending to the network
        metadata_clustering_models = loader.clustering_fit(train_metadata, ["RepetitionTime", "EchoTime", "FlipAngle"])
        train_datasets, train_onehotencoder = loader.normalize_metadata(train_datasets, metadata_clustering_models, context["debugging"], True)

    ds_train = ConcatDataset(train_datasets)
    print(f"Loaded {len(ds_train)} axial slices for the training set.")
    train_loader = DataLoader(ds_train, batch_size=context["batch_size"],
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=1)

    # Validation dataset ------------------------------------------------------
    validation_datasets = []
    for bids_ds in tqdm(context["bids_path_validation"], desc="Loading validation set"):
        ds_val = loader.BidsDataset(bids_ds,
                                    contrast_lst=context["contrast_train_validation"],
                                    transform=val_transform,
                                    slice_filter_fn=SliceFilter())
        validation_datasets.append(ds_val)

    if context["film"]:  # normalize metadata before sending to network
        validation_datasets = loader.normalize_metadata(validation_datasets, metadata_clustering_models, context["debugging"], False)

    ds_val = ConcatDataset(validation_datasets)
    print(f"Loaded {len(ds_val)} axial slices for the validation set.")
    val_loader = DataLoader(ds_val, batch_size=context["batch_size"],
                            shuffle=True, pin_memory=True,
                            collate_fn=mt_datasets.mt_collate,
                            num_workers=1)

    if context["film"]:
        # Modulated U-net model with FiLM layers
        model = models.FiLMedUnet(n_metadata=len([ll for l in train_onehotencoder.categories_ for ll in l]),
                            drop_rate=context["dropout_rate"],
                            bn_momentum=context["batch_norm_momentum"])
    else:
        # Traditional U-Net model
        model = models.Unet(drop_rate=context["dropout_rate"],
                            bn_momentum=context["batch_norm_momentum"])
    if cuda_available:
        model.cuda()

    num_epochs = context["num_epochs"]
    initial_lr = context["initial_lr"]

    # Using Adam with cosine annealing learning rate
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Write the metrics, images, etc to TensorBoard format
    writer = SummaryWriter(logdir=context["log_directory"])

    # Create dict containing gammas and betas after each FiLM layer.
    gammas_dict = {i:[] for i in range(1,9)}
    betas_dict = {i:[] for i in range(1,9)}

    # Create a list containing the contrast of all batch images
    var_contrast_list = []

    # Training loop -----------------------------------------------------------
    best_validation_loss = float("inf")
    for epoch in tqdm(range(1, num_epochs+1), desc="Training"):
        start_time = time.time()

        scheduler.step()

        lr = scheduler.get_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)

        model.train()
        train_loss_total = 0.0
        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples = batch["input"], batch["gt"]

            # The variable sample_metadata is where the MRI phyisics parameters are,
            # to get the metadata for the first sample for example, just use:
            # ---> bids_metadata_example = sample_metadata[0]["bids_metadata"]
            sample_metadata = batch["input_metadata"]

            if cuda_available:
                var_input = input_samples.cuda()
                var_gt = gt_samples.cuda(non_blocking=True)
            else:
                var_input = input_samples
                var_gt = gt_samples

            # var_contrast is the list of the batch sample's contrasts (eg T2w, T1w).
            var_contrast = [sample_metadata[k]['contrast'] for k in range(len(sample_metadata))]

            if context["film"]:
                var_metadata = [train_onehotencoder.transform([sample_metadata[k]['bids_metadata']]).tolist()[0] for k in range(len(sample_metadata))]
                preds = model(var_input, var_metadata)  # Input the metadata related to the input samples
            else:
                preds = model(var_input)

            loss = mt_losses.dice_loss(preds, var_gt)
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
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

        tqdm.write(f"Epoch {epoch} training loss: {train_loss_total_avg:.4f}.")

        # Validation loop -----------------------------------------------------
        model.eval()
        val_loss_total = 0.0
        num_steps = 0

        metric_fns = [mt_metrics.dice_score,
                      mt_metrics.hausdorff_score,
                      mt_metrics.precision_score,
                      mt_metrics.recall_score,
                      mt_metrics.specificity_score,
                      mt_metrics.intersection_over_union,
                      mt_metrics.accuracy_score]

        metric_mgr = mt_metrics.MetricManager(metric_fns)

        for i, batch in enumerate(val_loader):
            input_samples, gt_samples = batch["input"], batch["gt"]
            sample_metadata = batch["input_metadata"]

            with torch.no_grad():
                if cuda_available:
                    var_input = input_samples.cuda()
                    var_gt = gt_samples.cuda(non_blocking=True)
                else:
                    var_input = input_samples
                    var_gt = gt_samples

                # var_contrast is the list of the batch sample's contrasts (eg T2w, T1w).
                var_contrast = [sample_metadata[k]['contrast'] for k in range(len(sample_metadata))]

                if context["film"]:
                    var_metadata = [train_onehotencoder.transform([sample_metadata[k]['bids_metadata']]).tolist()[0] for k in range(len(sample_metadata))]
                    preds = model(var_input, var_metadata)  # Input the metadata related to the input samples
                else:
                    preds = model(var_input)

                loss = mt_losses.dice_loss(preds, var_gt)
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
            if epoch == num_epochs and i < int(len(ds_val)/context["batch_size"])+1:

                # Get all the contrasts of all batches
                var_contrast_list.append(var_contrast)

                # Fill the lists of gammas and betas
                gammas_dict[1].append(model.film1.gammas[:, :, 0, 0].cpu().numpy())
                betas_dict[1].append(model.film1.betas[:, :, 0, 0].cpu().numpy())

                gammas_dict[2].append(model.film2.gammas[:, :, 0, 0].cpu().numpy())
                betas_dict[2].append(model.film2.betas[:, :, 0, 0].cpu().numpy())

                gammas_dict[3].append(model.film3.gammas[:, :, 0, 0].cpu().numpy())
                betas_dict[3].append(model.film3.betas[:, :, 0, 0].cpu().numpy())

                gammas_dict[4].append(model.film4.gammas[:, :, 0, 0].cpu().numpy())
                betas_dict[4].append(model.film4.betas[:, :, 0, 0].cpu().numpy())

                gammas_dict[5].append(model.film5.gammas[:, :, 0, 0].cpu().numpy())
                betas_dict[5].append(model.film5.betas[:, :, 0, 0].cpu().numpy())

                gammas_dict[6].append(model.film6.gammas[:, :, 0, 0].cpu().numpy())
                betas_dict[6].append(model.film6.betas[:, :, 0, 0].cpu().numpy())

                gammas_dict[7].append(model.film7.gammas[:, :, 0, 0].cpu().numpy())
                betas_dict[7].append(model.film7.betas[:, :, 0, 0].cpu().numpy())

                gammas_dict[8].append(model.film8.gammas[:, :, 0, 0].cpu().numpy())
                betas_dict[8].append(model.film8.betas[:, :, 0, 0].cpu().numpy())

        metrics_dict = metric_mgr.get_results()
        metric_mgr.reset()

        writer.add_scalars('Validation/Metrics', metrics_dict, epoch)
        val_loss_total_avg = val_loss_total / num_steps
        writer.add_scalars('losses', {
            'train_loss': train_loss_total_avg,
            'val_loss': val_loss_total_avg,
        }, epoch)

        tqdm.write(f"Epoch {epoch} validation loss: {val_loss_total_avg:.4f}.")

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

        if val_loss_total_avg < best_validation_loss:
            best_validation_loss = val_loss_total_avg
            torch.save(model, "./"+context["log_directory"]+"/best_model.pt")

    # Save final model
    torch.save(model, "./"+context["log_directory"]+"/final_model.pt")
    if context["film"]:  # save clustering and OneHotEncoding models
        pickle.dump(metadata_clustering_models, open("./"+context["log_directory"]+"/clustering_models.pkl", 'wb'))
        pickle.dump(train_onehotencoder, open("./"+context["log_directory"]+"/one_hot_encoder.pkl", 'wb'))

    # Convert list of gammas/betas into numpy arrays
    gammas_dict = {i:np.array(gammas_dict[i]) for i in range(1,9)}
    betas_dict = {i:np.array(betas_dict[i]) for i in range(1,9)}

    # Save the numpy arrays for gammas/betas inside files.npy in log_directory
    param_outdir = os.path.join(context["log_directory"], "/film-parameters-visualization/film-parameters")
    if not os.path.isdir(param_outdir):
        os.makedirs(param_outdir)

    for i in range(1,9):
        np.save(param_outdir + f"/gamma_layer_{i}.npy", gammas_dict[i])
        np.save(param_outdir + f"/beta_layer_{i}.npy", betas_dict[i])

    # Convert into numpy and save the contrasts of all batch images
    contrast_images = np.array(var_contrast_list)
    np.save(context["log_directory"] + "/contrast_images.npy", contrast_images)

    return


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

    # These are the validation/testing transformations
    val_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((128, 128)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    test_datasets = []
    for bids_ds in tqdm(context["bids_path_test"], desc="Loading test set"):
        ds_test = loader.BidsDataset(bids_ds,
                                     contrast_lst=context["contrast_test"],
                                     transform=val_transform,
                                     slice_filter_fn=SliceFilter())
        test_datasets.append(ds_test)

    if context["film"]:  # normalize metadata before sending to network
        with open("./"+context["log_directory"]+"/clustering_models.pkl", 'rb') as file:
            metadata_clustering_models = pickle.load(file)
        test_datasets = loader.normalize_metadata(test_datasets, metadata_clustering_models, context["debugging"], False)

        with open("./"+context["log_directory"]+"/one_hot_encoder.pkl", 'rb') as file:
            one_hot_encoder = pickle.load(file)

    ds_test = ConcatDataset(test_datasets)
    print(f"Loaded {len(ds_test)} axial slices for the test set.")
    test_loader = DataLoader(ds_test, batch_size=context["batch_size"],
                             shuffle=True, pin_memory=True,
                             collate_fn=mt_datasets.mt_collate,
                             num_workers=1)

    model = torch.load("./"+context["log_directory"]+"/final_model.pt")

    if cuda_available:
        model.cuda()
    model.eval()

    metric_fns = [mt_metrics.dice_score,
                  # mt_metrics.hausdorff_score,
                  mt_metrics.precision_score,
                  mt_metrics.recall_score,
                  mt_metrics.specificity_score,
                  mt_metrics.intersection_over_union,
                  mt_metrics.accuracy_score]

    metric_mgr = mt_metrics.MetricManager(metric_fns)

    for i, batch in enumerate(test_loader):
        input_samples, gt_samples = batch["input"], batch["gt"]
        sample_metadata = batch["input_metadata"]

        with torch.no_grad():
            if cuda_available:
                test_input = input_samples.cuda()
                test_gt = gt_samples.cuda(non_blocking=True)
            else:
                test_input = input_samples
                test_gt = gt_samples

            test_contrast = [sample_metadata[k]['contrast'] for k in range(len(sample_metadata))]

            if context["film"]:
                test_metadata = [one_hot_encoder.transform([sample_metadata[k]['bids_metadata']]).tolist()[0] for k in range(len(sample_metadata))]
                preds = model(test_input, test_metadata)  # Input the metadata related to the input samples
            else:
                preds = model(test_input)

        # Metrics computation
        gt_npy = gt_samples.numpy().astype(np.uint8)
        gt_npy = gt_npy.squeeze(axis=1)

        preds_npy = preds.data.cpu().numpy()
        preds_npy = threshold_predictions(preds_npy)
        preds_npy = preds_npy.astype(np.uint8)

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
        shutil.copyfile(sys.argv[1], "./"+context["log_directory"]+"/config_file.json")
    elif command == 'test':
        cmd_test(context)

if __name__ == "__main__":
    run_main()

import sys
import json
import time

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
                    configuration file
    """
    # Set the GPU
    gpu_number = int(context["gpu"])
    torch.cuda.set_device(gpu_number)

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
                                      transform=train_transform,
                                      slice_filter_fn=SliceFilter())
        train_datasets.append(ds_train)
        train_metadata.append(ds_train.metadata)  # store the metadata of the training data, used for fitting the clustering models

    if context["film"]:  # normalize metadata before sending to the network
        metadata_clustering_models = loader.clustering_fit(train_metadata, ["RepetitionTime", "EchoTime", "FlipAngle"])
        train_datasets = loader.normalize_metadata(train_datasets, metadata_clustering_models, context["debugging"])

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
                                    transform=val_transform,
                                    slice_filter_fn=SliceFilter())
        validation_datasets.append(ds_val)

    if context["film"]:  # normalize metadata before sending to network
        validation_datasets = loader.normalize_metadata(validation_datasets, metadata_clustering_models, context["debugging"])

    ds_val = ConcatDataset(validation_datasets)
    print(f"Loaded {len(ds_val)} axial slices for the validation set.")
    val_loader = DataLoader(ds_val, batch_size=context["batch_size"],
                            shuffle=True, pin_memory=True,
                            collate_fn=mt_datasets.mt_collate,
                            num_workers=1)

    # Test dataset ------------------------------------------------------------
    test_datasets = []
    for bids_ds in tqdm(context["bids_path_test"], desc="Loading test set"):
        ds_test = loader.BidsDataset(bids_ds,
                                     transform=val_transform,
                                     slice_filter_fn=SliceFilter())
        test_datasets.append(ds_test)

    if context["film"]:  # normalize metadata before sending to network
        test_datasets = loader.normalize_metadata(test_datasets, metadata_clustering_models, context["debugging"])

    ds_test = ConcatDataset(test_datasets)
    print(f"Loaded {len(ds_test)} axial slices for the test set.")
    test_loader = DataLoader(ds_test, batch_size=context["batch_size"],
                             shuffle=True, pin_memory=True,
                             collate_fn=mt_datasets.mt_collate,
                             num_workers=1)

    if context["film"]:
        # Modulated U-net model with FiLM layers
        model = models.FiLMedUnet(drop_rate=context["dropout_rate"],
                            bn_momentum=context["batch_norm_momentum"])
    else:
        # Traditional U-Net model
        model = models.Unet(drop_rate=context["dropout_rate"],
                            bn_momentum=context["batch_norm_momentum"])
    model.cuda()

    num_epochs = context["num_epochs"]
    initial_lr = context["initial_lr"]

    # Using Adam with cosine annealing learning rate
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Write the metrics, images, etc to TensorBoard format
    writer = SummaryWriter(log_dir=context["log_directory"])

    # Training loop -----------------------------------------------------------
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

            var_input = input_samples.cuda()
            var_gt = gt_samples.cuda(non_blocking=True)
            var_metadata = sample_metadata

            if context["film"]:
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
            sample_metadata = batch["input_metadata"].cuda()

            with torch.no_grad():
                var_input = input_samples.cuda()
                var_gt = gt_samples.cuda(non_blocking=True)
                var_metadata = sample_metadata

                if context["film"]:
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

    return


def run_main():
    if len(sys.argv) <= 1:
        print("\nivadomed [config.json]\n")
        return

    with open(sys.argv[1], "r") as fhandle:
        context = json.load(fhandle)

    command = context["command"]

    if command == 'train':
        return cmd_train(context)


if __name__ == "__main__":
    run_main()

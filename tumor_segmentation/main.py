#
# To run in terminal:
# cd path/to/ivado-medical-imaging/tumor_segmentation
# python main.py config/config.json
#

import sys
import shutil
import json
import time
import torch
import joblib
import torchvision.utils as vutils
from bids_neuropoly import bids
from medicaltorch import datasets as mt_datasets
from medicaltorch import losses as mt_losses
from sklearn.model_selection import train_test_split
from ivadomed import loader
from ivadomed import models
from ivadomed.utils import *
from torch.utils.data import DataLoader
from torch import optim, nn
from tensorboardX import SummaryWriter
from tqdm import tqdm


def split_dataset(path_folder, random_seed, train_frac=0.6, test_frac=0.2):
    # read participants.tsv as pandas dataframe
    df = bids.BIDS(path_folder).participants.content
    # Separate dataset in test, train and validation using sklearn function
    X_train, X_remain = train_test_split(df['participant_id'].tolist(), train_size=train_frac, random_state=random_seed)
    X_test, X_val = train_test_split(X_remain, train_size=test_frac/(1 - train_frac), random_state=random_seed)
    return X_train, X_val, X_test

def cmd_train(context):
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

    training_set, val_set, test_set = split_dataset(context['bids_path'], random_seed=context['random_seed'])
    train_transforms = transforms.Compose([
        mt_transforms.CenterCrop2D((256, 256)),
        mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                       sigma_range=(3.5, 4.0),
                                       p=0.1),
        mt_transforms.RandomAffine(degrees=4.6,
                                   scale=(0.98, 1.02),
                                   translate=(0.03, 0.03)),
        mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    val_transforms = transforms.Compose([
        mt_transforms.CenterCrop2D((256, 256)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance()
    ])

    ds_train = loader.BidsDataset(context["bids_path"],
                                  subject_lst=training_set,
                                  gt_suffix=context["gt_suffix"],
                                  contrast_lst=context["contrast_train_validation"],
                                  contrast_balance=context["contrast_balance"],
                                  slice_axis=0,
                                  transform=train_transforms,
                                  slice_filter_fn=SliceFilter())

    print(f"Loaded {len(ds_train)} slices for the training set.")
    train_loader = DataLoader(ds_train, batch_size=context["batch_size"],
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=1)

    # Validation dataset ------------------------------------------------------
    ds_val = loader.BidsDataset(context["bids_path"],
                                subject_lst=val_set,
                                gt_suffix=context["gt_suffix"],
                                contrast_lst=context["contrast_train_validation"],
                                transform=val_transforms,
                                slice_axis=0,
                                slice_filter_fn=SliceFilter())
    print(f"Loaded {len(ds_val)} slices for the validation set.")
    val_loader = DataLoader(ds_val, batch_size=context["batch_size"],
                            shuffle=True, pin_memory=True,
                            collate_fn=mt_datasets.mt_collate,
                            num_workers=1)
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
    writer = SummaryWriter(log_dir=context["log_directory"])

    # Loss
    if context["loss"] in ["dice", "cross_entropy"]:
        if context["loss"] == "cross_entropy":
            loss_fct = nn.BCELoss()
    else:
        print("Unknown Loss function, please choose between 'dice' or 'cross_entropy'")
        exit()

    # Training loop -----------------------------------------------------------
    best_validation_loss = float("inf")
    bce_loss = nn.BCELoss()
    for epoch in tqdm(range(1, num_epochs+1), desc="Training"):
        start_time = time.time()

        lr = scheduler.get_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)

        model.train()
        train_loss_total = 0.0
        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples = batch["input"], batch["gt"]
            if cuda_available:
                var_input = input_samples.cuda()
                var_gt = gt_samples.cuda(non_blocking=True)
            else:
                var_input = input_samples
                var_gt = gt_samples
            preds = model(var_input)
            if context["loss"] == "dice":
                loss = mt_losses.dice_loss(preds, var_gt)
            else:
                loss = loss_fct(preds, var_gt)
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
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

            with torch.no_grad():
                if cuda_available:
                    var_input = input_samples.cuda()
                    var_gt = gt_samples.cuda(non_blocking=True)
                else:
                    var_input = input_samples
                    var_gt = gt_samples
                preds = model(var_input)
                if context["loss"] == "dice":
                    loss = mt_losses.dice_loss(preds, var_gt)
                else:
                    loss = loss_fct(preds, var_gt)
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

        if val_loss_total_avg < best_validation_loss:
            best_validation_loss = val_loss_total_avg
            torch.save(model, "./" + context["log_directory"] + "/best_model.pt")

        # Save final model
    torch.save(model, "./" + context["log_directory"] + "/final_model.pt")
    # save the subject distribution
    split_dct = {'train': training_set, 'valid': val_set, 'test': test_set}
    joblib.dump(split_dct, "./"+context["log_directory"]+"/split_datasets.joblib")

    writer.close()
    return

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

if __name__ == "__main__":
    run_main()
import sys
import json

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torch import optim

from medicaltorch import transforms as mt_transforms
from medicaltorch import datasets as mt_datasets
from medicaltorch import models as mt_models
from medicaltorch import losses as mt_losses
from medicaltorch import filters as mt_filters

from tensorboardX import SummaryWriter

from tqdm import tqdm

from ivadomed import loader as loader

cudnn.benchmark = True


def cmd_train(context):
    # Set the GPU
    gpu_number = int(context["gpu"])
    torch.cuda.set_device(gpu_number)

    train_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((200, 200)),
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

    train_datasets, train_metadata = [], []
    for bids_ds in context["bids_path_train"]:
        ds_train = loader.BidsDataset(bids_ds,
                                      transform=train_transform,
                                      slice_filter_fn=mt_filters.SliceFilter())
        train_datasets.append(ds_train)
        train_metadata.append(ds_train.metadata)
    ds_train = ConcatDataset(train_datasets)
    metadata_clustering_models = loader.clustering_fit(train_metadata, ["RepetitionTime", "EchoTime"])

    train_loader = DataLoader(ds_train, batch_size=context["batch_size"],
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=1)

    model = mt_models.Unet(drop_rate=context["dropout_rate"],
                           bn_momentum=context["batch_norm_momentum"])
    model.cuda()

    num_epochs = context["num_epochs"]
    initial_lr = context["initial_lr"]

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    writer = SummaryWriter(log_dir=context["log_directory"])

    for epoch in tqdm(range(1, num_epochs+1)):
        scheduler.step()

        lr = scheduler.get_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)

        model.train()
        train_loss_total = 0.0
        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples = batch["input"], batch["gt"]
            batch_metadata = batch["input_metadata"]
            # bids_metadata = batch_metadata["bids_metadata"]
            if context["normalize_metadata"]:
                batch_metadata = loader.normalize_metadata(batch_metadata, metadata_clustering_models, context["debugging"])
            
            var_input = input_samples.cuda()
            var_gt = gt_samples.cuda(non_blocking=True)

            preds = model(var_input)

            loss = mt_losses.dice_loss(preds, var_gt)
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

        train_loss_total_avg = train_loss_total / num_steps
        writer.add_scalars('losses', {
                             'train_loss': train_loss_total_avg
                           }, epoch)
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

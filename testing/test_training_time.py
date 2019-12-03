import numpy as np
import time

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from medicaltorch.filters import SliceFilter
from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms
from torch import optim

from tqdm import tqdm

from ivadomed import loader as loader
from ivadomed import models
from ivadomed import losses
from ivadomed.utils import *
import ivadomed.transforms as ivadomed_transforms

cudnn.benchmark = True

GPU_NUMBER = 5
BATCH_SIZE = 8
DROPOUT = 0.4
BN = 0.1
N_EPOCHS = 10
INIT_LR = 0.01
FILM_LAYERS = [0, 0, 0, 0, 0, 1, 1, 1]
PATH_BIDS = 'testing_data/'

def test_film_contrast(film_layers=FILM_LAYERS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(GPU_NUMBER)
        print("Using GPU number {}".format(GPU_NUMBER))

    training_transform_list = [
        ivadomed_transforms.Resample(wspace=0.75, hspace=0.75),
        ivadomed_transforms.ROICrop2D(size=[48, 48]),
        mt_transforms.ToTensor()
    ]
    train_transform = transforms.Compose(training_transform_list)

    train_lst = ['sub-test001']

    ds_train = loader.BidsDataset(PATH_BIDS,
                                  subject_lst=train_lst,
                                  target_suffix="_lesion-manual",
                                  roi_suffix="_seg-manual",
                                  contrast_lst=['T2w'],
                                  metadata_choice="contrast",
                                  contrast_balance={},
                                  slice_axis=2,
                                  transform=train_transform,
                                  multichannel=False,
                                  slice_filter_fn=SliceFilter(filter_empty_input=True, filter_empty_mask=False))

    ds_train.filter_roi(nb_nonzero_thr=10)

    metadata_clustering_models = None
    ds_train, train_onehotencoder = loader.normalize_metadata(ds_train,
                                                                metadata_clustering_models,
                                                                False,
                                                                "contrast",
                                                                True)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=0)

    model = models.FiLMedUnet(n_metadata=len([ll for l in train_onehotencoder.categories_ for ll in l]),
                               film_bool=film_layers,
                               drop_rate=DROPOUT,
                               bn_momentum=BN)
    if cuda_available:
        model.cuda()

    step_scheduler_batch = False
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        start_time = time.time()

        lr = scheduler.get_lr()[0]

        model.train()
        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples = batch["input"], batch["gt"]
            if cuda_available:
                var_input = input_samples.cuda()
                var_gt = gt_samples.cuda(non_blocking=True)

            sample_metadata = batch["input_metadata"]
            var_metadata = [train_onehotencoder.transform([sample_metadata[k]['film_input']]).tolist()[0] for k in range(len(sample_metadata))]
            preds = model(var_input, var_metadata)  # Input the metadata related to the input samples

            loss = - losses.dice_loss(preds, var_gt)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if step_scheduler_batch:
                scheduler.step()

            num_steps += 1

        if not step_scheduler_batch:
            scheduler.step()

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

def test_unet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(GPU_NUMBER)
        print("Using GPU number {}".format(GPU_NUMBER))

    training_transform_list = [
        ivadomed_transforms.Resample(wspace=0.75, hspace=0.75),
        ivadomed_transforms.ROICrop2D(size=[48, 48]),
        mt_transforms.ToTensor()
    ]
    train_transform = transforms.Compose(training_transform_list)

    train_lst = ['sub-test001']

    ds_train = loader.BidsDataset(PATH_BIDS,
                                  subject_lst=train_lst,
                                  target_suffix="_lesion-manual",
                                  roi_suffix="_seg-manual",
                                  contrast_lst=['T2w'],
                                  metadata_choice="without",
                                  contrast_balance={},
                                  slice_axis=2,
                                  transform=train_transform,
                                  multichannel=False,
                                  slice_filter_fn=SliceFilter(filter_empty_input=True, filter_empty_mask=False))

    ds_train.filter_roi(nb_nonzero_thr=10)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=0)

    model = models.Unet(
                        in_channel=1,
                        out_channel=1,
                        depth=2,
                        drop_rate=DROPOUT,
                        bn_momentum=BN)
    if torch.cuda.is_available():
        model.cuda()

    step_scheduler_batch = False
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        start_time = time.time()

        lr = scheduler.get_lr()[0]

        model.train()
        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples = batch["input"], batch["gt"]
            if cuda_available:
                var_input = input_samples.cuda()
                var_gt = gt_samples.cuda(non_blocking=True)

            preds = model(var_input)

            loss = - losses.dice_loss(preds, var_gt)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if step_scheduler_batch:
                scheduler.step()

            num_steps += 1

        if not step_scheduler_batch:
            scheduler.step()

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

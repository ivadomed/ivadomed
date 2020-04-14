import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from medicaltorch import transforms as mt_transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import ivadomed.transforms as ivadomed_transforms
from ivadomed import loader as imed_loader
from ivadomed import loader_utils as imed_loader_utils
from ivadomed import losses
from ivadomed import models
from ivadomed import utils as imed_utils

cudnn.benchmark = True

GPU_NUMBER = 0
BATCH_SIZE = 8
DROPOUT = 0.4
DEPTH = 3
BN = 0.1
N_EPOCHS = 10
INIT_LR = 0.01
FILM_LAYERS = [0, 0, 0, 0, 0, 1, 1, 1]
PATH_BIDS = 'testing_data'


def test_unet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(GPU_NUMBER)
        print("Using GPU number {}".format(GPU_NUMBER))

    training_transform_list = [
        ivadomed_transforms.Resample(wspace=0.75, hspace=0.75),
        ivadomed_transforms.ROICrop2D(size=[48, 48]),
        mt_transforms.ToTensor(),
        mt_transforms.StackTensors()
    ]
    training_transform_list_multichannel = training_transform_list.copy()
    training_transform_list_multichannel.append(mt_transforms.StackTensors())
    train_transform = transforms.Compose(training_transform_list)
    training_transform_multichannel = transforms.Compose(training_transform_list_multichannel)
    training_transform_3d_list = [
        ivadomed_transforms.CenterCrop3D(size=[96, 96, 16]),
        ivadomed_transforms.ToTensor3D(),
        mt_transforms.NormalizeInstance3D(),
        mt_transforms.StackTensors()
    ]

    train_lst = ['sub-test001']

    ds_train = imed_loader.BidsDataset(PATH_BIDS,
                                       subject_lst=train_lst,
                                       target_suffix=["_lesion-manual"],
                                       roi_suffix="_seg-manual",
                                       contrast_lst=['T2w'],
                                       metadata_choice="contrast",
                                       contrast_balance={},
                                       slice_axis=2,
                                       transform=train_transform,
                                       multichannel=False,
                                       slice_filter_fn=imed_utils.SliceFilter(filter_empty_input=True,
                                                                              filter_empty_mask=False))

    ds_mutichannel = imed_loader.BidsDataset(PATH_BIDS,
                                             subject_lst=train_lst,
                                             target_suffix=["_lesion-manual"],
                                             roi_suffix="_seg-manual",
                                             contrast_lst=['T1w', 'T2w'],
                                             metadata_choice="without",
                                             contrast_balance={},
                                             slice_axis=2,
                                             transform=training_transform_multichannel,
                                             multichannel=True,
                                             slice_filter_fn=imed_utils.SliceFilter(filter_empty_input=True,
                                                                                    filter_empty_mask=False))

    train_transform = transforms.Compose(training_transform_3d_list)
    ds_3d = imed_loader.Bids3DDataset(PATH_BIDS,
                                      subject_lst=train_lst,
                                      target_suffix=["_lesion-manual", "_seg-manual"],
                                      contrast_lst=['T1w', 'T2w'],
                                      metadata_choice="without",
                                      contrast_balance={},
                                      slice_axis=2,
                                      transform=train_transform,
                                      multichannel=False,
                                      length=[96, 96, 16],
                                      padding=0)

    ds_train = imed_loader_utils.filter_roi(ds_train, nb_nonzero_thr=10)

    metadata_clustering_models = None
    ds_train, train_onehotencoder = imed_loader_utils.normalize_metadata(ds_train,
                                                                         metadata_clustering_models,
                                                                         False,
                                                                         "contrast",
                                                                         True)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=imed_loader_utils.mt_collate,
                              num_workers=1)

    multichannel_loader = DataLoader(ds_mutichannel, batch_size=BATCH_SIZE,
                                     shuffle=True, pin_memory=True,
                                     collate_fn=imed_loader_utils.mt_collate,
                                     num_workers=1)
    loader_3d = DataLoader(ds_3d, batch_size=1,
                           shuffle=True, pin_memory=True,
                           collate_fn=imed_loader_utils.mt_collate,
                           num_workers=1)

    model_list = [(models.Unet(depth=DEPTH,
                               film_layers=FILM_LAYERS,
                               n_metadata=len(
                                   [ll for l in train_onehotencoder.categories_ for ll in l]),
                               drop_rate=DROPOUT,
                               bn_momentum=BN,
                               film_bool=True), train_loader, True, 'Filmed-Unet'),
                  (models.Unet(in_channel=1,
                               out_channel=1,
                               depth=2,
                               drop_rate=DROPOUT,
                               bn_momentum=BN), train_loader, False, 'Unet'),
                  (models.Unet(in_channel=2), multichannel_loader, False, 'Multi-Channels Unet'),
                  (models.UNet3D(in_channels=1, n_classes=2 + 1), loader_3d, False, '3D Unet'),
                  (models.UNet3D(in_channels=1, n_classes=2 + 1, attention=True), loader_3d, False, 'Attention UNet')]

    for model, train_loader, film_bool, model_name in model_list:
        print("Training {}".format(model_name))
        if cuda_available:
            model.cuda()

        step_scheduler_batch = False
        optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

        load_lst, pred_lst, opt_lst, schedul_lst, init_lst, gen_lst = [], [], [], [], [], []
        for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
            start_time = time.time()

            start_init = time.time()
            lr = scheduler.get_lr()[0]
            model.train()
            tot_init = time.time() - start_init
            init_lst.append(tot_init)

            num_steps = 0
            for i, batch in enumerate(train_loader):
                if i > 0:
                    tot_gen = time.time() - start_gen
                    gen_lst.append(tot_gen)

                start_load = time.time()
                input_samples, gt_samples = batch["input"], batch["gt"]
                if cuda_available:
                    var_input = input_samples.cuda()
                    var_gt = gt_samples.cuda(non_blocking=True)
                else:
                    var_input = input_samples
                    var_gt = gt_samples

                sample_metadata = batch["input_metadata"]
                if film_bool:
                    var_metadata = [train_onehotencoder.transform([sample_metadata[0][k]['film_input']]).tolist()[0]
                                    for k in range(len(sample_metadata[0]))]
                tot_load = time.time() - start_load
                load_lst.append(tot_load)

                start_pred = time.time()
                if film_bool:
                    # Input the metadata related to the input samples
                    preds = model(var_input, var_metadata)
                else:
                    preds = model(var_input)
                tot_pred = time.time() - start_pred
                pred_lst.append(tot_pred)

                start_opt = time.time()
                loss = - losses.dice_loss(preds, var_gt)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                if step_scheduler_batch:
                    scheduler.step()

                num_steps += 1
                tot_opt = time.time() - start_opt
                opt_lst.append(tot_opt)

                start_gen = time.time()

            start_schedul = time.time()
            if not step_scheduler_batch:
                scheduler.step()
            tot_schedul = time.time() - start_schedul
            schedul_lst.append(tot_schedul)

            end_time = time.time()
            total_time = end_time - start_time
            tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

        print('Mean SD init {} -- {}'.format(np.mean(init_lst), np.std(init_lst)))
        print('Mean SD load {} -- {}'.format(np.mean(load_lst), np.std(load_lst)))
        print('Mean SD pred {} -- {}'.format(np.mean(pred_lst), np.std(pred_lst)))
        print('Mean SDopt {} --  {}'.format(np.mean(opt_lst), np.std(opt_lst)))
        print('Mean SD gen {} -- {}'.format(np.mean(gen_lst), np.std(gen_lst)))
        print('Mean SD scheduler {} -- {}'.format(np.mean(schedul_lst), np.std(schedul_lst)))


print("Test training time")
test_unet()

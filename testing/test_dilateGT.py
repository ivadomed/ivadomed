import numpy as np
import time
import sys

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
PATH_BIDS = 'testing_data/'

def test_dilateGT():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(GPU_NUMBER)
        print("Using GPU number {}".format(GPU_NUMBER))

    training_transform_list = [
        ivadomed_transforms.Resample(wspace=0.75, hspace=0.75),
        ivadomed_transforms.ROICrop2D(size=[48, 48]),
        ivadomed_transforms.ToTensor()
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
                              num_workers=1)

    model = models.Unet(
                        in_channel=1,
                        out_channel=1,
                        depth=2,
                        drop_rate=DROPOUT,
                        bn_momentum=BN)
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
            tot_load = time.time() - start_load
            load_lst.append(tot_load)

            start_pred = time.time()
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
    print('Mean SD opt {} --  {}'.format(np.mean(opt_lst), np.std(opt_lst)))
    print('Mean SD gen {} -- {}'.format(np.mean(gen_lst), np.std(gen_lst)))
    print('Mean SD scheduler {} -- {}'.format(np.mean(schedul_lst), np.std(schedul_lst)))


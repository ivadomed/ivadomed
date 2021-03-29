import os
import time
import pytest
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import ivadomed.transforms as imed_transforms
from ivadomed import losses
from ivadomed import models
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, adaptative as imed_adaptative
from ivadomed import training as imed_training
import logging
from testing.unit_tests.t_utils import create_tmp_dir, __data_testing_dir__, __tmp_dir__, get_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir

logger = logging.getLogger(__name__)

cudnn.benchmark = True

GPU_ID = 0
BATCH_SIZE = 4
DROPOUT = 0.4
BN = 0.1
N_EPOCHS = 10
INIT_LR = 0.01
p = 0.0001
__path_hdf5__ = os.path.join(__data_testing_dir__, "mytestfile.hdf5")
__path_csv__ = os.path.join(__data_testing_dir__, "hdf5.csv")


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [__data_testing_dir__],
    "target_suffix": ["_lesion-manual"],
    "extensions": [".nii.gz"],
    "roi_params": {"suffix": "_seg-manual", "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": ['T1w', 'T2w', 'T2star']
                        }}])
@pytest.mark.run(order=1)
def test_HeMIS(get_data_testing_test_files, loader_parameters, p=0.0001, ):
    print('[INFO]: Starting test ... \n')

    bids_df = imed_loader_utils.BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)

    contrast_params = loader_parameters["contrast_params"]
    target_suffix = loader_parameters["target_suffix"]
    roi_params = loader_parameters["roi_params"]

    training_transform_dict = {
        "Resample":
            {
                "wspace": 0.75,
                "hspace": 0.75
            },
        "CenterCrop":
            {
                "size": [48, 48]
            },
        "NumpyToTensor": {}
    }

    transform_lst, _ = imed_transforms.prepare_transforms(training_transform_dict)

    train_lst = ['sub-unf01']

    print('[INFO]: Creating dataset ...\n')
    model_params = {
        "name": "HeMISUnet",
        "dropout_rate": 0.3,
        "bn_momentum": 0.9,
        "depth": 2,
        "in_channel": 1,
        "out_channel": 1,
        "missing_probability": 0.00001,
        "missing_probability_growth": 0.9,
        "contrasts": ["T1w", "T2w"],
        "ram": False,
        "path_hdf5": __path_hdf5__,
        "csv_path": __path_csv__,
        "target_lst": ["T2w"],
        "roi_lst": ["T2w"]
    }
    dataset = imed_adaptative.HDF5Dataset(bids_df=bids_df,
                                          subject_file_lst=train_lst,
                                          model_params=model_params,
                                          contrast_params=contrast_params,
                                          target_suffix=target_suffix,
                                          slice_axis=2,
                                          transform=transform_lst,
                                          metadata_choice=False,
                                          dim=2,
                                          slice_filter_fn=imed_loader_utils.SliceFilter(
                                              filter_empty_input=True,
                                              filter_empty_mask=True),
                                          roi_params=roi_params)

    dataset.load_into_ram(['T1w', 'T2w', 'T2star'])
    print("[INFO]: Dataset RAM status:")
    print(dataset.status)
    print("[INFO]: In memory Dataframe:")
    print(dataset.dataframe)

    # TODO
    # ds_train.filter_roi(nb_nonzero_thr=10)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=1)

    model = models.HeMISUnet(contrasts=contrast_params["contrast_lst"],
                             depth=3,
                             drop_rate=DROPOUT,
                             bn_momentum=BN)

    print(model)
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        torch.cuda.set_device(GPU_ID)
        print("Using GPU ID {}".format(GPU_ID))
        model.cuda()

    # Initialing Optimizer and scheduler
    step_scheduler_batch = False
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

    load_lst, reload_lst, pred_lst, opt_lst, schedul_lst, init_lst, gen_lst = [], [], [], [], [], [], []

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        start_time = time.time()

        start_init = time.time()
        lr = scheduler.get_last_lr()[0]
        model.train()

        tot_init = time.time() - start_init
        init_lst.append(tot_init)

        num_steps = 0
        start_gen = 0
        for i, batch in enumerate(train_loader):
            if i > 0:
                tot_gen = time.time() - start_gen
                gen_lst.append(tot_gen)

            start_load = time.time()
            input_samples, gt_samples = imed_utils.unstack_tensors(batch["input"]), batch["gt"]

            print(batch["input_metadata"][0][0]["missing_mod"])
            missing_mod = imed_training.get_metadata(batch["input_metadata"], model_params)

            print("Number of missing contrasts = {}."
                  .format(len(input_samples) * len(input_samples[0]) - missing_mod.sum()))
            print("len input = {}".format(len(input_samples)))
            print("Batch = {}, {}".format(input_samples[0].shape, gt_samples[0].shape))

            if cuda_available:
                var_input = imed_utils.cuda(input_samples)
                var_gt = imed_utils.cuda(gt_samples, non_blocking=True)
            else:
                var_input = input_samples
                var_gt = gt_samples

            tot_load = time.time() - start_load
            load_lst.append(tot_load)

            start_pred = time.time()
            preds = model(var_input, missing_mod)
            tot_pred = time.time() - start_pred
            pred_lst.append(tot_pred)

            start_opt = time.time()
            loss = - losses.DiceLoss()(preds, var_gt)

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

        start_reload = time.time()
        print("[INFO]: Updating Dataset")
        p = p ** (2 / 3)
        dataset.update(p=p)
        print("[INFO]: Reloading dataset")
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory=True,
                                  collate_fn=imed_loader_utils.imed_collate,
                                  num_workers=1)
        tot_reload = time.time() - start_reload
        reload_lst.append(tot_reload)

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

    print('Mean SD init {} -- {}'.format(np.mean(init_lst), np.std(init_lst)))
    print('Mean SD load {} -- {}'.format(np.mean(load_lst), np.std(load_lst)))
    print('Mean SD reload {} -- {}'.format(np.mean(reload_lst), np.std(reload_lst)))
    print('Mean SD pred {} -- {}'.format(np.mean(pred_lst), np.std(pred_lst)))
    print('Mean SD opt {} --  {}'.format(np.mean(opt_lst), np.std(opt_lst)))
    print('Mean SD gen {} -- {}'.format(np.mean(gen_lst), np.std(gen_lst)))
    print('Mean SD scheduler {} -- {}'.format(np.mean(schedul_lst), np.std(schedul_lst)))


@pytest.mark.run(order=2)
def test_hdf5_bids(get_data_testing_test_files):
    __output_dir__ = os.path.join(__tmp_dir__, "test_adap_bids")
    os.makedirs(__output_dir__)
    imed_adaptative.HDF5ToBIDS(
        __path_hdf5__,
        ['sub-unf01'],
        __output_dir__)
    assert os.path.isdir(os.path.join(__output_dir__, "sub-unf01/anat"))
    assert os.path.isdir(os.path.join(__output_dir__, "derivatives/labels/sub-unf01/anat"))
    print('\n [INFO]: Test of HeMIS passed successfully.')


def teardown_function():
    remove_tmp_dir()

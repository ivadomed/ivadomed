from ivadomed import adaptative as adaptative
from medicaltorch.filters import SliceFilter

from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms
import os
from ivadomed import loader as loader
from ivadomed import models
from ivadomed import losses
from ivadomed.utils import *
import ivadomed.transforms as ivadomed_transforms

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

GPU_NUMBER = 0
BATCH_SIZE = 8
DROPOUT = 0.4
DEPTH = 3
BN = 0.1
N_EPOCHS = 10
INIT_LR = 0.01
FILM_LAYERS = [0, 0, 0, 0, 0, 1, 1, 1]
PATH_BIDS = 'testing_data'


def test_hdf5():
    print('[INFO]: Starting test ... \n')
    train_lst = ['sub-test001']

    hdf5_file = adaptative.Bids_to_hdf5(PATH_BIDS,
                                        subject_lst=train_lst,
                                        hdf5_name='testing_data/mytestfile.hdf5',
                                        target_suffix="_lesion-manual",
                                        roi_suffix="_seg-manual",
                                        contrast_lst=['T1w', 'T2w', 'T2star'],
                                        metadata_choice="contrast",
                                        contrast_balance={},
                                        slice_axis=2,
                                        slice_filter_fn=SliceFilter(filter_empty_input=True, filter_empty_mask=True))

    # Checking architecture
    def print_attrs(name, obj):
        print("\nName of the object: {}".format(name))
        print("Type: {}".format(type(obj)))
        print("Including the following attributes:")
        for key, val in obj.attrs.items():
            print("    %s: %s" % (key, val))

    #hdf5_file.hdf5_file.visititems(print_attrs)
    print('\n[INFO]: HDF5 file successfully generated.')
    print('[INFO]: Generating dataframe ...\n')
    
    df = adaptative.Dataframe(hdf5=hdf5_file.hdf5_file,
                              contrasts=['T1w', 'T2w', 'T2star'],
                              path='testing_data/hdf5.csv',
                              target_suffix=['T1w', 'T2w', 'T2star'],
                              roi_suffix=['T1w', 'T2w', 'T2star'],
                              dim=2,
                              slices=True)
    print(df.df)

    print('\n[INFO]: Dataframe successfully generated. ')
    print('[INFO]: Creating dataset ...\n')

    dataset = adaptative.HDF5Dataset(root_dir=PATH_BIDS,
                                     subject_lst=train_lst,
                                     hdf5_name='testing_data/mytestfile.hdf5',
                                     csv_name='testing_data/hdf5.csv',
                                     target_suffix="_lesion-manual",
                                     contrast_lst=['T1w', 'T2w', 'T2star'],
                                     ram=False,
                                     contrast_balance={},
                                     slice_axis=2,
                                     transform=None,
                                     metadata_choice=False,
                                     dim=2,
                                     slice_filter_fn=SliceFilter(filter_empty_input=True, filter_empty_mask=True),
                                     canonical=True,
                                     roi_suffix="_seg-manual",
                                     target_lst=['T2w'],
                                     roi_lst=['T2w'])

    dataset.load_into_ram(['T1w', 'T2w', 'T2star'])
    print("Dataset RAM status:")
    print(dataset.status)
    print("In memory Dataframe:")
    print(dataset.dataframe)
    print('\n[INFO]: Test passed successfully. ')

    print("\n[INFO]: Starting loader test ...")

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

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=1)
    model = models.HeMISUnet(modalities=contrasts,
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
            print("len input = {}".format(len(input_samples)))
            print("Batch = {}, {}".format(input_samples[0].shape, gt_samples.shape))

            print("Congrats your dataloader works!")
            return 0

test_hdf5()

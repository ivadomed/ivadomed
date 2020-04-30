import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as torch_transforms

import ivadomed.transforms as imed_transforms
from ivadomed.loader import utils as imed_loader_utils, adaptative as imed_adaptative
from ivadomed import utils as imed_utils

GPU_NUMBER = 0
BATCH_SIZE = 4
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

    hdf5_file = imed_adaptative.Bids_to_hdf5(PATH_BIDS,
                                             subject_lst=train_lst,
                                             hdf5_name='testing_data/mytestfile.hdf5',
                                             target_suffix=["_lesion-manual"],
                                             roi_suffix="_seg-manual",
                                             contrast_lst=['T1w', 'T2w', 'T2star'],
                                             metadata_choice="contrast",
                                             contrast_balance={},
                                             slice_axis=2,
                                             slice_filter_fn=imed_utils.SliceFilter(filter_empty_input=True,
                                                                                    filter_empty_mask=True))

    # Checking architecture
    def print_attrs(name, obj):
        print("\nName of the object: {}".format(name))
        print("Type: {}".format(type(obj)))
        print("Including the following attributes:")
        for key, val in obj.attrs.items():
            print("    %s: %s" % (key, val))

    print('\n[INFO]: HDF5 architecture:')
    hdf5_file.hdf5_file.visititems(print_attrs)
    print('\n[INFO]: HDF5 file successfully generated.')
    print('[INFO]: Generating dataframe ...\n')

    df = imed_adaptative.Dataframe(hdf5=hdf5_file.hdf5_file,
                                   contrasts=['T1w', 'T2w', 'T2star'],
                                   path='testing_data/hdf5.csv',
                                   target_suffix=['T1w', 'T2w', 'T2star'],
                                   roi_suffix=['T1w', 'T2w', 'T2star'],
                                   dim=2,
                                   filter_slices=True)

    print(df.df)

    print('\n[INFO]: Dataframe successfully generated. ')
    print('[INFO]: Creating dataset ...\n')

    training_transform_list = [
        imed_transforms.Resample(wspace=0.75, hspace=0.75),
        imed_transforms.CenterCrop2D(size=[48, 48]),
        imed_transforms.ToTensor()
    ]
    train_transform = torch_transforms.Compose(training_transform_list)

    dataset = imed_adaptative.HDF5Dataset(root_dir=PATH_BIDS,
                                          subject_lst=train_lst,
                                          hdf5_name='testing_data/mytestfile.hdf5',
                                          csv_name='testing_data/hdf5.csv',
                                          target_suffix="_lesion-manual",
                                          contrast_lst=['T1w', 'T2w', 'T2star'],
                                          ram=False,
                                          contrast_balance={},
                                          slice_axis=2,
                                          transform=train_transform,
                                          metadata_choice=False,
                                          dim=2,
                                          slice_filter_fn=imed_utils.SliceFilter(filter_empty_input=True,
                                                                                 filter_empty_mask=True),
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

    device = torch.device("cuda:"+str(GPU_NUMBER) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(device)
        print("Using GPU number {}".format(device))

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              shuffle=False, pin_memory=True,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=1)

    for i, batch in enumerate(train_loader):
        input_samples, gt_samples = batch["input"], batch["gt"]
        print("len input = {}".format(len(input_samples)))
        print("Batch = {}, {}".format(input_samples[0].shape, gt_samples[0].shape))

        if cuda_available:
            var_input = imed_utils.cuda(input_samples)
            var_gt = imed_utils.cuda(gt_samples, non_blocking=True)
        else:
            var_input = input_samples
            var_gt = gt_samples

        break
    os.remove('testing_data/mytestfile.hdf5')
    print("Congrats your dataloader works! You can go Home now and get a beer.")
    return 0


test_hdf5()

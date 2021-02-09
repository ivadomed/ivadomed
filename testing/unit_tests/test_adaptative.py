import os
import h5py
import torch
from torch.utils.data import DataLoader

import ivadomed.transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, adaptative as imed_adaptative
import logging
from unit_tests.t_utils import remove_tmp_dir, create_tmp_dir, __data_testing_dir__
logger = logging.getLogger(__name__)

GPU_ID = 0
BATCH_SIZE = 4
DROPOUT = 0.4
DEPTH = 3
BN = 0.1
N_EPOCHS = 10
INIT_LR = 0.01
FILM_LAYERS = [0, 0, 0, 0, 0, 1, 1, 1]


def setup_function():
    create_tmp_dir()


def test_hdf5():
    print('[INFO]: Starting test ... \n')
    train_lst = ['sub-unf01']

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

    roi_params = {"suffix": "_seg-manual", "slice_filter_roi": None}

    bids_to_hdf5 = imed_adaptative.BIDStoHDF5(root_BIDS_list=[__data_testing_dir__],
                                              subject_lst=train_lst,
                                              path_hdf5=os.path.join(__data_testing_dir__, 'mytestfile.hdf5'),
                                              target_suffix=["_lesion-manual"],
                                              roi_params=roi_params,
                                              contrast_lst=['T1w', 'T2w', 'T2star'],
                                              metadata_choice="contrast",
                                              transform=transform_lst,
                                              contrast_balance={},
                                              slice_axis=2,
                                              slice_filter_fn=imed_loader_utils.SliceFilter(
                                                filter_empty_input=True,
                                                filter_empty_mask=True))

    # Checking architecture
    def print_attrs(name, obj):
        print("\nName of the object: {}".format(name))
        print("Type: {}".format(type(obj)))
        print("Including the following attributes:")
        for key, val in obj.attrs.items():
            print("    %s: %s" % (key, val))

    print('\n[INFO]: HDF5 architecture:')
    with h5py.File(bids_to_hdf5.path_hdf5, "a") as hdf5_file:
        hdf5_file.visititems(print_attrs)
        print('\n[INFO]: HDF5 file successfully generated.')
        print('[INFO]: Generating dataframe ...\n')

        df = imed_adaptative.Dataframe(hdf5_file=hdf5_file,
                                       contrasts=['T1w', 'T2w', 'T2star'],
                                       path=os.path.join(__data_testing_dir__, 'hdf5.csv'),
                                       target_suffix=['T1w', 'T2w', 'T2star'],
                                       roi_suffix=['T1w', 'T2w', 'T2star'],
                                       dim=2,
                                       filter_slices=True)

        print(df.df)

        print('\n[INFO]: Dataframe successfully generated. ')
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
                "path_hdf5": os.path.join(__data_testing_dir__, 'mytestfile.hdf5'),
                "csv_path": os.path.join(__data_testing_dir__, 'hdf5.csv'),
                "target_lst": ["T2w"],
                "roi_lst": ["T2w"]
            }
        contrast_params = {
            "contrast_lst": ['T1w', 'T2w', 'T2star'],
            "balance": {}
        }

        dataset = imed_adaptative.HDF5Dataset(root_dir=[__data_testing_dir__],
                                              subject_lst=train_lst,
                                              target_suffix="_lesion-manual",
                                              slice_axis=2,
                                              model_params=model_params,
                                              contrast_params=contrast_params,
                                              transform=transform_lst,
                                              metadata_choice=False,
                                              dim=2,
                                              slice_filter_fn=imed_loader_utils.SliceFilter(
                                                filter_empty_input=True,
                                                filter_empty_mask=True),
                                              roi_params=roi_params)

        dataset.load_into_ram(['T1w', 'T2w', 'T2star'])
        print("Dataset RAM status:")
        print(dataset.status)
        print("In memory Dataframe:")
        print(dataset.dataframe)
        print('\n[INFO]: Test passed successfully. ')

        print("\n[INFO]: Starting loader test ...")

        device = torch.device("cuda:" + str(GPU_ID) if torch.cuda.is_available() else "cpu")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.set_device(device)
            print("Using GPU ID {}".format(device))

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
        print("Congrats your dataloader works! You can go home now and get a beer.")
        return 0


def teardown_function():
    remove_tmp_dir()

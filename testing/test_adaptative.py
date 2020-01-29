from ivadomed import adaptative as adaptative
from medicaltorch.filters import SliceFilter
import os

GPU_NUMBER = 5
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

    hdf5_file.hdf5_file.visititems(print_attrs)
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
                                     slice_filter_fn=None,
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


test_hdf5()

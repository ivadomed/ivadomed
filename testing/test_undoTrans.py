import numpy as np
from random import randint

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from ivadomed.utils import SliceFilter
from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms

from ivadomed import loader as loader
import ivadomed.transforms as ivadomed_transforms

cudnn.benchmark = True

GPU_NUMBER = 5
SLICE_AXIS = 2
PATH_BIDS = 'testing_data'


def test_undo(contrast='T2star', tol=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        torch.cuda.set_device(GPU_NUMBER)
        print("using GPU number {}".format(GPU_NUMBER))

    # reference
    test_1 = [ivadomed_transforms.ToTensor(), ivadomed_transforms.NormalizeInstance()]
    name_1 = 'ToTensor_NoramlizeInstance'

    test_2 = [mt_transforms.RandomRotation(degrees=10)] + test_1
    name_2 = 'RandomRotation_' + name_1

    test_3 = [mt_transforms.CenterCrop2D(size=[40, 48])] + test_2
    name_3 = 'CenterCrop2D_' + name_2

    test_4 = [ivadomed_transforms.ROICrop2D(size=[40, 48])] + test_2
    name_4 = 'ROICrop2D_' + name_2

    test_5 = [ivadomed_transforms.Resample(wspace=0.75, hspace=0.75)] + test_3
    name_5 = 'Resample_' + name_3

    test_6 = [ivadomed_transforms.Resample(wspace=0.75, hspace=0.75)] + test_4
    name_6 = 'Resample_' + name_4

    name_lst = [name_2, name_3, name_4, name_5, name_6]
    test_lst = [test_2, test_3, test_4, test_5, test_6]

    subject_test_lst = ['sub-test001']

    ds_test_noTrans = loader.BidsDataset(PATH_BIDS,
                                         subject_lst=subject_test_lst,
                                         target_suffix="_lesion-manual",
                                         roi_suffix="_seg-manual",
                                         contrast_lst=[contrast],
                                         metadata_choice="contrast",
                                         contrast_balance={},
                                         slice_axis=SLICE_AXIS,
                                         transform=transforms.Compose(test_1),
                                         multichannel=False,
                                         slice_filter_fn=SliceFilter(filter_empty_input=True,
                                                                     filter_empty_mask=False))
    test_loader_noTrans = DataLoader(ds_test_noTrans, batch_size=len(ds_test_noTrans),
                                     shuffle=False, pin_memory=True,
                                     collate_fn=mt_datasets.mt_collate,
                                     num_workers=1)
    batch_noTrans = [t for i, t in enumerate(test_loader_noTrans)][0]
    input_noTrans, gt_noTrans = batch_noTrans["input"], batch_noTrans["gt"]

    for name, test in zip(name_lst, test_lst):
        print('\n [INFO]: Test of {} ... '.format(name))
        val_transform = transforms.Compose(test)
        val_undo_transform = ivadomed_transforms.UndoCompose(val_transform)

        ds_test = loader.BidsDataset(PATH_BIDS,
                                     subject_lst=subject_test_lst,
                                     target_suffix="_lesion-manual",
                                     roi_suffix="_seg-manual",
                                     contrast_lst=[contrast],
                                     metadata_choice="contrast",
                                     contrast_balance={},
                                     slice_axis=SLICE_AXIS,
                                     transform=val_transform,
                                     multichannel=False,
                                     slice_filter_fn=SliceFilter(filter_empty_input=True,
                                                                 filter_empty_mask=False))

        test_loader = DataLoader(ds_test, batch_size=len(ds_test),
                                 shuffle=False, pin_memory=True,
                                 collate_fn=mt_datasets.mt_collate,
                                 num_workers=1)
        batch = [t for i, t in enumerate(test_loader)][0]

        input_, gt_ = batch["input"], batch["gt"]

        for smp_idx in range(len(batch['gt'])):
            # re-load all dict
            rdict = {}
            for k in batch.keys():
                rdict[k] = batch[k][smp_idx]

            # undo transformations
            rdict_undo = val_undo_transform(rdict)

            # get data
            np_noTrans = np.array(gt_noTrans[smp_idx])[0]
            np_undoTrans = np.array(rdict_undo['gt'])

            # binarise
            np_undoTrans[np_undoTrans > 0] = 1.0

            # check shapes
            print(np_noTrans.shape, np_undoTrans.shape)
            assert np_noTrans.shape == np_undoTrans.shape
            print('\tData shape: checked.')

            # check values for ROICrop
            if np.any(np_noTrans) and not 'CenterCrop2D' in name:
                # if difference is superior to tolerance, then save images to QC
                if np.sum(np_noTrans-np_undoTrans) >= tol:
                    print(np.sum(np_noTrans-np_undoTrans))
                    im_noTrans = np.array(input_noTrans[smp_idx])[0]
                    im_undoTrans = np.array(rdict_undo['input'])

                    plt.figure(figsize=(20, 10))
                    plt.subplot(1, 2, 1)
                    plt.axis("off")
                    plt.imshow(im_noTrans, interpolation='nearest', aspect='auto', cmap='gray')
                    plt.subplot(1, 2, 2)
                    plt.axis("off")
                    plt.imshow(im_undoTrans, interpolation='nearest', aspect='auto', cmap='gray')

                    fname_png_out = 'test_undo_err_'+str(randint(0, 1000))+'.png'
                    plt.savefig(fname_png_out, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    print('Error: please check: '+fname_png_out)

                assert np.sum(np_noTrans-np_undoTrans) < tol
                print('\tData content (tol: {} vox.): checked.'.format(tol))
        print('\n [INFO]: Test of {} passed successfully. '.format(name))


print("Test undo transform")
test_undo()

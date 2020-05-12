import torch
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import loader as imed_loader, utils as imed_loader_utils
from torchvision import transforms as torch_transforms
from torch.utils.data import DataLoader
import numpy as np

import nibabel as nib

GPU_NUMBER = 0
PATH_BIDS = 'testing_data'


def test_image_orientation():
    device = torch.device("cuda:" + str(GPU_NUMBER) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(device)
        print("Using GPU number {}".format(device))

    train_lst = ['sub-test001']

    training_transform_list = [
        imed_transforms.Resample(hspace=2, wspace=2),
        imed_transforms.CenterCrop2D(size=[96, 96]),
        imed_transforms.ToTensor(),
        imed_transforms.NormalizeInstance(),
    ]
    training_transform = torch_transforms.Compose(training_transform_list)
    training_undo_transform = imed_transforms.UndoCompose(training_transform)

    for slice_axis in [0, 1, 2]:
        ds = imed_loader.BidsDataset(PATH_BIDS,
                                     subject_lst=train_lst,
                                     target_suffix=["_seg-manual"],
                                     contrast_lst=['T1w'],
                                     metadata_choice="without",
                                     contrast_balance={},
                                     slice_axis=slice_axis,
                                     transform=training_transform,
                                     multichannel=False)

        loader = DataLoader(ds, batch_size=1,
                            shuffle=True, pin_memory=True,
                            collate_fn=imed_loader_utils.imed_collate,
                            num_workers=1)

        input_filename, gt_filename, roi_filename, metadata = ds.filename_pairs[0]
        segpair = imed_loader.SegmentationPair(input_filename, gt_filename, metadata=metadata)
        nib_original = nib.load(input_filename[0])
        input_init = nib_original.get_fdata()
        input_ras = nib.as_closest_canonical(nib_original).get_fdata()
        img, gt = segpair.get_pair_data()
        input_hwd = img[0]

        pred_tmp_lst, z_tmp_lst = [], []
        for i, batch in enumerate(loader):
            input_samples, gt_samples = batch["input"], batch["gt"]

            batch["input_metadata"] = batch["input_metadata"][0]  # Take only metadata from one input
            batch["gt_metadata"] = batch["gt_metadata"][0]  # Take only metadata from one label

            for smp_idx in range(len(batch['gt'])):
                # undo transformations
                rdict = {}
                for k in batch.keys():
                    rdict[k] = batch[k][smp_idx]
                rdict_undo = training_undo_transform(rdict)

                # add new sample to pred_tmp_lst
                pred_tmp_lst.append(imed_utils.pil_list_to_numpy(rdict_undo['input']))
                z_tmp_lst.append(int(rdict_undo['input_metadata']['slice_index']))
                fname_ref = rdict_undo['input_metadata']['gt_filenames'][0]

                if pred_tmp_lst and i == len(loader) - 1:
                    # save the completely processed file as a nii
                    nib_ref = nib.load(fname_ref)
                    nib_ref_can = nib.as_closest_canonical(nib_ref)

                    tmp_lst = []
                    for z in range(nib_ref_can.header.get_data_shape()[slice_axis]):
                        if not z in z_tmp_lst:
                            tmp_lst.append(np.zeros(pred_tmp_lst[0].shape))
                        else:
                            tmp_lst.append(pred_tmp_lst[z_tmp_lst.index(z)])

                    # create data and stack on depth dimension
                    input_hwd_2 = np.stack(tmp_lst, axis=-1)
                    input_ras_2 = imed_loader_utils.orient_img_ras(nib_ref_can.get_fdata())
                    input_init_2 = imed_utils.reorient_image(input_hwd_2, slice_axis, nib_ref, nib_ref_can)

                    # re-init pred_stack_lst
                    pred_tmp_lst, z_tmp_lst = [], []


print("Test image orientation")
test_image_orientation()

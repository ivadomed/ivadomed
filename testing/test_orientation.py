import torch
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import loader as imed_loader, utils as imed_loader_utils
from torchvision import transforms as torch_transforms
from torch.utils.data import DataLoader

import nibabel as nib

GPU_NUMBER = 0
PATH_BIDS = 'testing_data'
SLICE_AXIS = [0, 1, 2]


def test_image_orientation():
    device = torch.device("cuda:" + str(GPU_NUMBER) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(device)
        print("Using GPU number {}".format(device))

    train_lst = ['sub-test001']

    training_transform_3d_list = [
        imed_transforms.CenterCrop3D(size=[96, 96, 16]),
        imed_transforms.ToTensor3D(),
        imed_transforms.NormalizeInstance3D(),
        imed_transforms.StackTensors()
    ]
    training_transform_3d = torch_transforms.Compose(training_transform_3d_list)
    training_undo_transform = imed_transforms.UndoCompose(training_transform_3d_list)

    ds_3d = imed_loader.Bids3DDataset(PATH_BIDS,
                                      subject_lst=train_lst,
                                      target_suffix=["_seg-manual"],
                                      contrast_lst=['T1w', 'T2w'],
                                      metadata_choice="without",
                                      contrast_balance={},
                                      slice_axis=2,
                                      transform=training_transform_3d,
                                      multichannel=False,
                                      length=[96, 96, 16],
                                      padding=0)

    loader_3d = DataLoader(ds_3d, batch_size=1,
                           shuffle=True, pin_memory=True,
                           collate_fn=imed_loader_utils.imed_collate,
                           num_workers=1)

    input_filename, gt_filename, roi_filename, metadata = ds_3d.filename_pairs[0]
    segpair = imed_loader.SegmentationPair(input_filename, gt_filename, metadata=metadata)
    nib_original = nib.load(input_filename)
    input_init = nib_original.get_fdata()
    input_ras = nib.as_closest_canonical(nib_original).get_fdata()
    input_hwd, gt = segpair.get_pair_data()

    for i, batch in enumerate(loader_3d):
        input_samples, gt_samples = batch["input"], batch["gt"]

        for smp_idx in range(len(batch['gt'])):
            # undo transformations
            rdict = {}
            for k in batch.keys():
                rdict[k] = batch[k][smp_idx]
            rdict_undo = training_undo_transform(rdict)
            input_hwd_2 = rdict_undo['input']

            fname_ref = rdict_undo['input_metadata']['gt_filenames'][0]
            nib_ref = nib.load(fname_ref)
            nib_ref_can = nib.as_closest_canonical(nib_ref)
            input_ras_2 = nib_ref_can.get_fdata()
            input_init_2 = imed_utils.reorient_image(input_hwd_2, SLICE_AXIS, nib_ref, nib_ref_can)



print("Test image orientation")
test_image_orientation()
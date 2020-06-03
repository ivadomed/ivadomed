import os
import pytest
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from ivadomed import metrics as imed_metrics
from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader

cudnn.benchmark = True

GPU_NUMBER = 0
BATCH_SIZE = 8
DROPOUT = 0.4
BN = 0.1
SLICE_AXIS = 2
PATH_BIDS = 'testing_data'
PATH_OUT = 'tmp'

@pytest.mark.parametrize('transforms_dict', [{
        "Resample":
            {
                "wspace": 0.75,
                "hspace": 0.75
            },
        "CenterCrop":
            {
                "size": [48, 48]
            },
        "NumpyToTensor": {},
        "NormalizeInstance": {"applied_to": ["im"]}
    }])
@pytest.mark.parametrize('test_lst', [['sub-test001']])
@pytest.mark.parametrize('target_lst', [["_seg-manual"]])
@pytest.mark.parametrize('roi_params', [{"suffix": None, "slice_filter_roi": None}])
def test_inference(transforms_dict, test_lst, target_lst, roi_params):
    cuda_available, device = imed_utils.define_device(GPU_NUMBER)

    loader_params = {
        "transforms_params": transforms_dict,
        "data_list": test_lst,
        "dataset_type": "testing",
        "requires_undo": True,
        "contrast_lst": ['T2w', 'T2star'],
        "balance": {},
        "bids_path": PATH_BIDS,
        "target_suffix": target_lst,
        "roi_params": roi_params,
        "slice_filter_params": {
            "filter_empty_mask": False,
            "filter_empty_input": True
        },
        "slice_axis": SLICE_AXIS,
        "multichannel": False
    }

    # Get Testing dataset
    ds_test = imed_loader.load_dataset(**loader_params)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=0)

    # Undo transform
    val_undo_transform = imed_transforms.UndoCompose(imed_transforms.Compose(transforms_dict))

    # Model
    model_path = os.path.join(PATH_BIDS, "model_unet_test.pt")
    model = torch.load(model_path, map_location=device)

    if cuda_available:
        model.cuda()
    model.eval()

    metric_fns = [imed_metrics.dice_score,
                  imed_metrics.hausdorff_2D_score,
                  imed_metrics.precision_score,
                  imed_metrics.recall_score,
                  imed_metrics.specificity_score,
                  imed_metrics.intersection_over_union,
                  imed_metrics.accuracy_score]

    metric_mgr = imed_metrics.MetricManager(metric_fns)

    if not os.path.isdir(PATH_OUT):
        os.makedirs(PATH_OUT)

    pred_tmp_lst, z_tmp_lst, fname_tmp = [], [], ''
    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            input_samples = imed_utils.cuda(batch["input"], cuda_available)
            gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)

            preds = model(input_samples)

        preds_cpu = preds.cpu()

        # reconstruct 3D image
        for smp_idx in range(len(batch['gt'])):
            # undo transformations
            preds_idx_undo, metadata_idx = val_undo_transform(preds_cpu[smp_idx], batch["gt_metadata"][smp_idx],
                                                              data_type='gt')

            # preds_idx_undo is a list of length n_label of arrays
            preds_idx_arr = np.array(preds_idx_undo)

            # TODO: gt_filenames should not be a list
            fname_ref = metadata_idx[0]['gt_filenames'][0]
            # new processed file
            if pred_tmp_lst and (
                    fname_ref != fname_tmp or (i == len(test_loader) - 1 and smp_idx == len(batch['gt']) - 1)):
                # save the completely processed file as a nii
                fname_pred = os.path.join(PATH_OUT, fname_tmp.split('/')[-1])
                fname_pred = fname_pred.split('manual.nii.gz')[0] + 'pred.nii.gz'
                _ = imed_utils.pred_to_nib(pred_tmp_lst, z_tmp_lst, fname_tmp, fname_pred, SLICE_AXIS,
                                           debug=True, kernel_dim='2d', bin_thr=0.5)

                # re-init pred_stack_lst
                pred_tmp_lst, z_tmp_lst = [], []

            # add new sample to pred_tmp_lst
            pred_tmp_lst.append(preds_idx_arr)
            z_tmp_lst.append(int(batch['input_metadata'][smp_idx][0]['slice_index']))
            fname_tmp = fname_ref

        # Metrics computation
        gt_npy = gt_samples.numpy().astype(np.uint8)
        gt_npy = gt_npy.squeeze(axis=1)

        preds_npy = preds.data.cpu().numpy()
        preds_npy = imed_postpro.threshold_predictions(preds_npy)
        preds_npy = preds_npy.astype(np.uint8)
        preds_npy = preds_npy.squeeze(axis=1)

        metric_mgr(preds_npy, gt_npy)

    metrics_dict = metric_mgr.get_results()
    metric_mgr.reset()
    print(metrics_dict)

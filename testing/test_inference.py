import os

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


def test_inference(film_bool=False):
    device = torch.device("cuda:" + str(GPU_NUMBER) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        pin_memory = False
        print("cuda is not available.")
        print("Working on {}.".format("cpu"))
    if cuda_available:
        pin_memory = True
        # Set the GPU
        torch.cuda.set_device(device)
        print("using GPU number {}".format(device))

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
        "NumpyToTensor": {},
        "NormalizeInstance": {"applied_to": ["im"]}
    }

    val_transform = imed_transforms.Compose(training_transform_dict)

    val_undo_transform = imed_transforms.UndoCompose(val_transform)

    test_lst = ['sub-test001']
    contrast_params = {
        "contrast_lst": ['T2w', 'T2star'],
        "balance": {}
    }
    ds_test = imed_loader.BidsDataset(PATH_BIDS,
                                      subject_lst=test_lst,
                                      target_suffix=["_lesion-manual"],
                                      roi_suffix="_seg-manual",
                                      contrast_params=contrast_params,
                                      metadata_choice="contrast",
                                      slice_axis=SLICE_AXIS,
                                      transform=val_transform,
                                      multichannel=False,
                                      slice_filter_fn=imed_utils.SliceFilter(filter_empty_input=True,
                                                                             filter_empty_mask=False))

    ds_test = imed_loader_utils.filter_roi(ds_test, nb_nonzero_thr=10)

    if film_bool:  # normalize metadata before sending to network
        print('FiLM inference not implemented yet.')
        return 0
        # metadata_clustering_models = joblib.load("./" + context["log_directory"] + "/clustering_models.joblib")
        # ds_test = imed_film.normalize_metadata(ds_test,
        #                                     metadata_clustering_models,
        #                                     context["debugging"],
        #                                     context["metadata"],
        #                                     False)

        # one_hot_encoder = joblib.load("./" + context["log_directory"] + "/one_hot_encoder.joblib")

    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=pin_memory,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=1)

    model = torch.load(os.path.join(PATH_BIDS, "model_unet_test.pt"),
                       map_location=device)

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
        input_samples, gt_samples = batch["input"], batch["gt"]

        with torch.no_grad():
            if cuda_available:
                test_input = imed_utils.cuda(input_samples)
                test_gt = imed_utils.cuda(gt_samples, non_blocking=True)
            else:
                test_input = input_samples
                test_gt = gt_samples

            preds = model(test_input)

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

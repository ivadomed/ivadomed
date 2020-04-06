import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from ivadomed import metrics

from medicaltorch import datasets as mt_datasets

from ivadomed import loader as loader
import ivadomed.postprocessing as imed_postPro
import ivademed.utils as imed_utils
import ivadomed.transforms as imed_transforms

cudnn.benchmark = True

GPU_NUMBER = 7
BATCH_SIZE = 8
DROPOUT = 0.4
BN = 0.1
SLICE_AXIS = 2
PATH_BIDS = 'testing_data'
PATH_OUT = 'tmp'


def test_inference(film_bool=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        pin_memory = False
        print("cuda is not available.")
        print("Working on {}.".format("cpu"))
    if cuda_available:
        pin_memory = True
        # Set the GPU
        torch.cuda.set_device(GPU_NUMBER)
        print("using GPU number {}".format(GPU_NUMBER))

    validation_transform_list = [
        imed_transforms.Resample(wspace=0.75, hspace=0.75),
        imed_transforms.ROICrop2D(size=[48, 48]),
        imed_transforms.ToTensor(),
        imed_transforms.NormalizeInstance()
    ]

    val_transform = transforms.Compose(validation_transform_list)
    val_undo_transform = imed_transforms.UndoCompose(val_transform)

    test_lst = ['sub-test001']

    ds_test = loader.BidsDataset(PATH_BIDS,
                                 subject_lst=test_lst,
                                 target_suffix="_lesion-manual",
                                 roi_suffix="_seg-manual",
                                 contrast_lst=['T2w', 'T2star'],
                                 metadata_choice="contrast",
                                 contrast_balance={},
                                 slice_axis=SLICE_AXIS,
                                 transform=val_transform,
                                 multichannel=False,
                                 slice_filter_fn=imed_utils.SliceFilter(filter_empty_input=True,
                                                                         filter_empty_mask=False))

    ds_test = loader.filter_roi(ds_test, nb_nonzero_thr=10)

    if film_bool:  # normalize metadata before sending to network
        print('FiLM inference not implemented yet.')
        return 0
        # metadata_clustering_models = joblib.load("./" + context["log_directory"] + "/clustering_models.joblib")
        # ds_test = loader.normalize_metadata(ds_test,
        #                                     metadata_clustering_models,
        #                                     context["debugging"],
        #                                     context["metadata"],
        #                                     False)

        # one_hot_encoder = joblib.load("./" + context["log_directory"] + "/one_hot_encoder.joblib")

    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=pin_memory,
                             collate_fn=mt_datasets.mt_collate,
                             num_workers=1)

    if film_bool:
        model = torch.load(os.path.join(PATH_BIDS, "model_film_test.pt"), map_location=device)
    else:
        model = torch.load(os.path.join(PATH_BIDS, "model_unet_test.pt"), map_location=device)

    if cuda_available:
        model.cuda()
    model.eval()

    metric_fns = [metrics.dice_score,  # from ivadomed/utils.py
                  metrics.hausdorff_2D_score,
                  metrics.precision_score,
                  metrics.recall_score,
                  metrics.specificity_score,
                  metrics.intersection_over_union,
                  metrics.accuracy_score]

    metric_mgr = metrics.MetricManager(metric_fns)

    if not os.path.isdir(PATH_OUT):
        os.makedirs(PATH_OUT)

    pred_tmp_lst, z_tmp_lst, fname_tmp = [], [], ''
    for i, batch in enumerate(test_loader):
        input_samples, gt_samples = batch["input"], batch["gt"]

        with torch.no_grad():
            if cuda_available:
                test_input = input_samples.cuda()
                test_gt = gt_samples.cuda(non_blocking=True)
            else:
                test_input = input_samples
                test_gt = gt_samples

            if film_bool:
                sample_metadata = batch["input_metadata"]
                test_contrast = [sample_metadata[k]['contrast']
                                 for k in range(len(sample_metadata))]

                test_metadata = [one_hot_encoder.transform([sample_metadata[k]["film_input"]]).tolist()[0] for k in
                                 range(len(sample_metadata))]
                # Input the metadata related to the input samples
                preds = model(test_input, test_metadata)
            else:
                preds = model(test_input)

        # WARNING: sample['gt'] is actually the pred in the return sample
        # implementation justification: the other option: rdict['pred'] = preds would require to largely modify mt_transforms
        rdict = {}
        rdict['gt'] = preds.cpu()
        batch.update(rdict)

        # reconstruct 3D image
        for smp_idx in range(len(batch['gt'])):
            # undo transformations
            rdict = {}
            for k in batch.keys():
                rdict[k] = batch[k][smp_idx]
            rdict_undo = val_undo_transform(rdict)

            fname_ref = rdict_undo['input_metadata']['gt_filename']
            # new processed file
            if pred_tmp_lst and (fname_ref != fname_tmp or (i == len(test_loader)-1 and smp_idx == len(batch['gt'])-1)):
                # save the completely processed file as a nii
                fname_pred = os.path.join(PATH_OUT, fname_tmp.split('/')[-1])
                fname_pred = fname_pred.split('manual.nii.gz')[0] + 'pred.nii.gz'
                _ = imed_utils.pred_to_nib(pred_tmp_lst, z_tmp_lst, fname_tmp, fname_pred, SLICE_AXIS, debug=True, kernel_dim='2d', bin_thr=0.5)
                # re-init pred_stack_lst
                pred_tmp_lst, z_tmp_lst = [], []

            # add new sample to pred_tmp_lst
            pred_tmp_lst.append(np.array(rdict_undo['gt']))
            z_tmp_lst.append(int(rdict_undo['input_metadata']['slice_index']))
            fname_tmp = fname_ref

        # Metrics computation
        gt_npy = gt_samples.numpy().astype(np.uint8)
        gt_npy = gt_npy.squeeze(axis=1)

        preds_npy = preds.data.cpu().numpy()
        preds_npy = imed_postPro.threshold_predictions(preds_npy)
        preds_npy = preds_npy.astype(np.uint8)
        preds_npy = preds_npy.squeeze(axis=1)

        metric_mgr(preds_npy, gt_npy)

    metrics_dict = metric_mgr.get_results()
    metric_mgr.reset()
    print(metrics_dict)


print("Test unet")
test_inference()
print("test unet-filmed")
# test_inference(film_bool=True)

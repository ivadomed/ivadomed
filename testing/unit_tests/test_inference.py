import os
import pytest
import shutil
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from ivadomed import metrics as imed_metrics
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed import testing as imed_testing
from ivadomed import models as imed_models
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader
import logging
from t_utils import remove_tmp_dir, create_tmp_dir, __data_testing_dir__, __tmp_dir__
logger = logging.getLogger(__name__)

cudnn.benchmark = True

GPU_ID = 0
BATCH_SIZE = 8
DROPOUT = 0.4
BN = 0.1
SLICE_AXIS = "axial"
__output_dir__ = os.path.join(__tmp_dir__, "output_inference")


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('transforms_dict', [{
        "Resample": {
                "wspace": 0.75,
                "hspace": 0.75
            },
        "CenterCrop": {
                "size": [48, 48]
            },
        "NumpyToTensor": {},
        "NormalizeInstance": {"applied_to": ["im"]}
    }])
@pytest.mark.parametrize('test_lst', [['sub-unf01']])
@pytest.mark.parametrize('target_lst', [["_lesion-manual"], ["_seg-manual"]])
@pytest.mark.parametrize('roi_params', [{"suffix": "_seg-manual", "slice_filter_roi": 10}])
@pytest.mark.parametrize('testing_params', [{
    "binarize_prediction": 0.5,
    "uncertainty": {
        "applied": False,
        "epistemic": False,
        "aleatoric": False,
        "n_it": 0
    }}])
def test_inference(transforms_dict, test_lst, target_lst, roi_params, testing_params):
    cuda_available, device = imed_utils.define_device(GPU_ID)

    model_params = {"name": "Unet", "is_2d": True}
    loader_params = {
        "transforms_params": transforms_dict,
        "data_list": test_lst,
        "dataset_type": "testing",
        "requires_undo": True,
        "contrast_params": {"contrast_lst": ['T2w'], "balance": {}},
        "bids_path": __data_testing_dir__,
        "target_suffix": target_lst,
        "roi_params": roi_params,
        "slice_filter_params": {
            "filter_empty_mask": False,
            "filter_empty_input": True
        },
        "slice_axis": SLICE_AXIS,
        "multichannel": False
    }
    loader_params.update({"model_params": model_params})

    # Get Testing dataset
    ds_test = imed_loader.load_dataset(**loader_params)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=0)

    # Undo transform
    val_undo_transform = imed_transforms.UndoCompose(imed_transforms.Compose(transforms_dict))

    # Update testing_params
    testing_params.update({
        "slice_axis": loader_params["slice_axis"],
        "target_suffix": loader_params["target_suffix"],
        "undo_transforms": val_undo_transform
    })

    # Model
    model = imed_models.Unet()

    if cuda_available:
        model.cuda()
    model.eval()

    metric_fns = [imed_metrics.dice_score,
                  imed_metrics.hausdorff_score,
                  imed_metrics.precision_score,
                  imed_metrics.recall_score,
                  imed_metrics.specificity_score,
                  imed_metrics.intersection_over_union,
                  imed_metrics.accuracy_score]

    metric_mgr = imed_metrics.MetricManager(metric_fns)

    if not os.path.isdir(__output_dir__):
        os.makedirs(__output_dir__)

    preds_npy, gt_npy = imed_testing.run_inference(test_loader=test_loader,
                                                   model=model,
                                                   model_params=model_params,
                                                   testing_params=testing_params,
                                                   ofolder=__output_dir__,
                                                   cuda_available=cuda_available)

    metric_mgr(preds_npy, gt_npy)
    metrics_dict = metric_mgr.get_results()
    metric_mgr.reset()
    print(metrics_dict)


def teardown_function():
    remove_tmp_dir()

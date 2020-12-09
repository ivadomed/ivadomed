import ivadomed.utils as imed_utils
import ivadomed.visualize as imed_visualize
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os
import io
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import ivadomed.maths as imed_math
from PIL import Image
import time
import shutil


def test_tensorboard_save():
    inp = torch.tensor(np.zeros((1, 1, 15, 15)))
    gt = torch.tensor(np.zeros((1, 1, 15, 15)))
    pred = torch.tensor(np.zeros((1, 1, 15, 15)))
    dpath = "test_tensorboard_save"
    os.makedirs(dpath)
    writer = SummaryWriter(log_dir=dpath)
    imed_visualize.save_tensorboard_img(writer, 1, "Training", inp, pred, gt)
    writer.flush()

    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
    for i in range(len(summary_iterators)):
        if summary_iterators[i].Tags()['images'] == ['Training/Input', 'Training/Predictions', 'Training/Ground Truth']:
            input_retrieve = np.array(Image.open(io.BytesIO(summary_iterators[i].Images('Training/Input')[0][2])))
            pred_retrieve = np.array(Image.open(io.BytesIO(summary_iterators[i].Images('Training/Predictions')[0][2])))
            gt_retrieve = np.array(Image.open(io.BytesIO(summary_iterators[i].Images('Training/Ground Truth')[0][2])))

    assert np.allclose(imed_math.rescale_values_array(input_retrieve[:, :, 0], 0, 1), inp[0, 0, :, :])
    assert np.allclose(imed_math.rescale_values_array(pred_retrieve[:, :, 0], 0, 1), pred[0, 0, :, :])
    assert np.allclose(imed_math.rescale_values_array(gt_retrieve[:, :, 0], 0, 1), gt[0, 0, :, :])
    shutil.rmtree(dpath)

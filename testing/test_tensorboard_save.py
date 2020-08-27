import ivadomed.utils as imed_utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os
import io
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from PIL import Image


def test_tensorboard_save():
    inp = torch.tensor(np.zeros((1, 1, 15, 15)))
    gt = torch.tensor(np.zeros((1, 1, 15, 15)))
    pred = torch.tensor(np.zeros((1, 1, 15, 15)))
    dpath = "test_tensorboard_save"
    os.makedirs(dpath)
    writer = SummaryWriter(log_dir=dpath)
    imed_utils.save_tensorboard_img(writer, 1, "Training", inp, pred, gt)

    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
    for i in range(len(summary_iterators)):
        if summary_iterators[i].Tags()['images'] == ['Training/Input', 'Training/Predictions', 'Training/Ground Truth']:
            input_retrieve = np.array(Image.open(io.BytesIO(summary_iterators[i].Images('Training/Input')[0][2])))
            pred_retrieve = np.array(Image.open(io.BytesIO(summary_iterators[i].Images('Training/Predictions')[0][2])))
            gt_retrieve = np.array(Image.open(io.BytesIO(summary_iterators[i].Images('Training/Ground Truth')[0][2])))

    assert np.allclose(input_retrieve, inp)
    assert np.allclose(pred_retrieve, pred)
    assert np.allclose(gt_retrieve, gt)



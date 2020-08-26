import ivadomed.utils as imed_utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os


def test_tensorboard_save():
    inp = torch.tensor(np.zeros((1, 1, 15, 15)))
    gt = torch.tensor(np.zeros((1, 1, 15, 15)))
    pred = torch.tensor(np.zeros((1, 1, 15, 15)))
    os.makedirs("test_tensorboard_save")
    writer = SummaryWriter(log_dir="test_tensorboard_save")
    imed_utils.save_tensorboard_img(writer, 1, "Training", inp, pred, gt)
    


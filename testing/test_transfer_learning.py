import sys
import json
import os

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim

from ivadomed import models

cudnn.benchmark = True

GPU_NUMBER = 5
N_METADATA = 1
INITIAL_LR = 0.001
FILM_LAYERS = [0, 0, 0, 0, 0, 0, 0, 0]
PATH_PRETRAINED_MODEL = 'testing_data/model_unet_test.pt'

def test_transfer_learning(film_layers=FILM_LAYERS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("Cuda is not available.")
        print("Working on {}.".format('cpu'))
    if cuda_available:
        # Set the GPU
        torch.cuda.set_device(GPU_NUMBER)
        print("Using GPU number {}".format(GPU_NUMBER))

    film_bool = bool(sum(film_layers))

    if film_bool:
        n_metadata = N_METADATA

    # Traditional U-Net model
    in_channel = 1



    if cuda_available:
        model.cuda()

    initial_lr = INITIAL_LR

    # Using Adam
    step_scheduler_batch = False
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

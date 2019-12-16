import sys
import json
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim

from ivadomed import models

cudnn.benchmark = True

GPU_NUMBER = 1
N_METADATA = 1
OUT_CHANNEL = 1
INITIAL_LR = 0.001
FILM_LAYERS = [0, 0, 0, 0, 0, 0, 0, 0]
PATH_PRETRAINED_MODEL = 'testing_data/model_unet_test.pt'

def test_transfer_learning(film_layers=FILM_LAYERS, path_model=PATH_PRETRAINED_MODEL):
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

    model = torch.load(path_model)

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last conv layer
    # Note: Parameters of newly constructed layer have requires_grad=True by default
    model.decoder.last_conv = nn.Conv2d(model.decoder.last_conv.in_channels,
                                        OUT_CHANNEL, kernel_size=3, padding=1)

    if film_bool and film_layers[-1]:
        model.decoder.last_film = models.FiLMlayer(n_metadata, 1)

    if cuda_available:
        model.cuda()

    print('Layers included in the optimisation:')
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    assert(total_params > total_trainable_params)

    initial_lr = INITIAL_LR
    params_to_opt = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_opt, lr=initial_lr)

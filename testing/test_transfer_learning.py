import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim

from ivadomed import models as imed_models

cudnn.benchmark = True

GPU_NUMBER = 7
N_METADATA = 1
OUT_CHANNEL = 1
INITIAL_LR = 0.001
FILM_LAYERS = [0, 0, 0, 0, 0, 0, 0, 0]
PATH_PRETRAINED_MODEL = 'testing_data/model_unet_test.pt'
RETRAIN_FRACTION = 0.3

def test_transfer_learning(film_layers=FILM_LAYERS, path_model=PATH_PRETRAINED_MODEL, fraction=RETRAIN_FRACTION):
    device = torch.device("cuda:"+str(GPU_NUMBER) if torch.cuda.is_available() else "cpu")
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

    model = torch.load(path_model, map_location=device)

    model = imed_models.set_model_for_retrain(model, fraction)

    if cuda_available:
        model.cuda()

    print('Layers included in the optimisation:')
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    assert(total_params > total_trainable_params)

    initial_lr = INITIAL_LR
    params_to_opt = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_opt, lr=initial_lr)


print("test transfer learning")
test_transfer_learning()

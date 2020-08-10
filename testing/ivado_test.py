#Add custom tests here
# this test is run first so we will also do a little setup here

import ivadomed.models as imed_models
import torch
import os


def test_sample():
    assert 1 == 1


def test_model_creation():
    # creating basic model for test
    model = imed_models.Unet()
    torch.save(model, "testing_data/model_unet_test.pt")
    assert os.path.isfile("testing_data/model_unet_test.pt")


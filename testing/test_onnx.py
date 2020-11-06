import os
import nibabel as nib
import torch
import numpy as np
import json
import shutil

from ivadomed import utils as imed_utils
from ivadomed import inference as imed_inference
from ivadomed import models as imed_models


PATH_BIDS = 'testing_data'
IMAGE_PATH = os.path.join(PATH_BIDS, "sub-unf01", "anat", "sub-unf01_T1w.nii.gz")
PATH_MODEL = os.path.join(PATH_BIDS, 'model')
PATH_MODEL_ONNX = os.path.join(PATH_MODEL, 'model.onnx')
PATH_MODEL_PT = PATH_MODEL_ONNX.replace('onnx', 'pt')
LENGTH_3D = (112, 112, 112)


def test_onnx():
    model = imed_models.UNet3D(1, 1)
    if not os.path.exists(PATH_MODEL):
        os.mkdir(PATH_MODEL)
    torch.save(model, PATH_MODEL_PT)
    img = nib.load(IMAGE_PATH).get_fdata().astype('float32')[:16, :64, :32]
    # Add batch and channel dimensions
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    dummy_input = torch.randn(1, 1, 32, 32, 32)
    imed_utils.save_onnx_model(model, dummy_input, PATH_MODEL_ONNX)

    model = torch.load(PATH_MODEL_PT)
    model.eval()
    out_pt = model(img_tensor).detach().numpy()

    out_onnx = imed_inference.onnx_inference(PATH_MODEL_ONNX, img_tensor).numpy()
    shutil.rmtree(PATH_MODEL)
    assert np.allclose(out_pt, out_onnx, rtol=1e-3)

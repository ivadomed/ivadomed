from ivadomed.utils import segment_volume, save_onnx_model
from ivadomed import postprocessing as imed_postpro
from ivadomed import models as imed_models
import psutil, os
import nibabel as nib
from memory_profiler import memory_usage
import torch

PATH = "/home/andreanne/Documents/models/sc_model_30mm"
# PATH = "/home/andreanne/Downloads/t2star_sc"

# m = imed_models.UNet3D(1, 1)
# m.load_state_dict(torch.load(os.path.join(PATH, "t2star_sc.pt")))

# torch.jit.save(imed_models.Unet, os.path.join(PATH, "t2star_sc2.pt"))
# torch.save(model.state_dict(), os.path.join(PATH, "t2star_sc.pt"))

def test():
    nii = segment_volume(PATH, "/home/andreanne/Documents/dataset/pad_512x256x16/data/sub-Astr144/anat/sub-Astr144_T2w.nii.gz")
    imed_postpro.threshold_predictions(nii, 0.5)
    nib.save(nii, "/home/andreanne/Documents/models/sc_model_30mm/out.nii.gz")

# model = torch.load(os.path.join(PATH, 'sc_model_30mm.pt'), map_location="cuda:0")
# model.eval()
# dummy_input = torch.randn(2, 1, 96, 96, 16, device='cuda:0')
# save_onnx_model(model, os.path.join(PATH, 'sc_model_30mm.pt').replace('pt', 'onnx'),  dummy_input)
print(max(memory_usage(test)))
# path = "/home/andreanne/Documents/dataset/tumor_segmentation_masks/results/data/derivatives/labels/sub-Astr144/anat/sub-Astr144_T2w_sc-mask.nii.gz"
# mask = nib.load(path).get_fdata()
# imed_postpro.generate_bounding_box(mask)
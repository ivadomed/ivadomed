from ivadomed.utils import segment_volume
from ivadomed import postprocessing as imed_postpro
from ivadomed import models as imed_models
import psutil, os
import nibabel as nib
import torch

PATH = "/home/andreanne/Downloads/t2star_sc"

# m = imed_models.Unet()
# m.load_state_dict(torch.load(os.path.join(PATH, "t2star_sc.pt")))

# torch.jit.save(imed_models.Unet, os.path.join(PATH, "t2star_sc2.pt"))
# torch.save(model.state_dict(), os.path.join(PATH, "t2star_sc.pt"))


nii = segment_volume("/home/andreanne/Documents/models/seg_tumor", "/home/andreanne/Documents/dataset/pad_512x256x16/data/sub-Astr144/anat/sub-Astr144_T2w.nii.gz")
imed_postpro.threshold_predictions(nii, 0.5)
nib.save(nii, "/home/andreanne/Documents/out.nii.gz")

# path = "/home/andreanne/Documents/dataset/tumor_segmentation_masks/results/data/derivatives/labels/sub-Astr144/anat/sub-Astr144_T2w_sc-mask.nii.gz"
# mask = nib.load(path).get_fdata()
# imed_postpro.generate_bounding_box(mask)
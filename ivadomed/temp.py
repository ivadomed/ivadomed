import os
path_to_img = "/home/andreanne/Documents/dataset/tumor_segmentation_masks/results/data/sub-Astr144/anat/sub-Astr144_T2w.nii.gz"
model1 = "/home/andreanne/Documents/models/sc_model_30mm_2"
model2 = "/home/andreanne/Documents/models/t2_tumor"

os.system(f"sct_deepseg -i {path_to_img} -model {model1}")
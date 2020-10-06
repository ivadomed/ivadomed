import os
import random
import shutil

path = "../duke/sct_testing/gmseg_challenge_16/"
subfolders = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

for subfolder in subfolders:

    if "ucl" in subfolder:
        sub_no = subfolder.split("_")[1]
        folder_name = "sub-ucl" + sub_no
    elif "unf_pain" in subfolder:
        sub_no = subfolder.split("_")[2]
        folder_name = "sub-unf" + sub_no
    elif "vanderbilt" in subfolder:
        sub_no = subfolder.split("_")[1]
        folder_name = "sub-vanderbilt" + sub_no
    elif "zurich" in subfolder:
        sub_no = subfolder.split("_")[1]
        folder_name = "sub-zurich" + sub_no
    else:
        print("erreur")

    files = os.listdir(os.path.join(path,subfolder,"t2s"))
    niis = [file for file in files if ("nii.gz" in file)]
    if len(niis) != 5:
        print("error not 5 files in folder", subfolder)
    print(niis)
    #for nii in niis:

    print(sub_no)
    print(folder_name)

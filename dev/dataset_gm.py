import os
import random
import shutil

path = "~/duke/sct_testing/gmseg_challenge_16"
subfolders = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

for subfolder in subfolders:

    if "ucl" in subfolder:
        sub_no = subfolder.split("_")[2]
        folder_name = "sub-ucl" + sub_no
    if "unf_pain" in subfolder:
        sub_no = subfolder.split("_")[1]
        folder_name = "sub-unf" + sub_no
    elif "vanderbilt" in subfolder:
        sub_no = subfolder.split("_")[1]
        folder_name = "sub-vanderbilt" + sub_no
    elif "zurich" in subfolder:
        sub_no = subfolder.split("_")[1]
        folder_name = "sub-zurich" + sub_no


    else:
        print("erreur")

    print(sub_no)
    print(folder_name)

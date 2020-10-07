import os
import random
import shutil

path = "../duke/sct_testing/gmseg_challenge_16/"
path2 = "../duke/projects/ivadomed/gm_challenge_16_inter_rater/"
subfolders = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
subfolders2 = [ name for name in os.listdir(path2) if os.path.isdir(os.path.join(path2, name)) ]
centers = {"ucl":"1", "unf":"2", "vanderbilt":"3", "zurich":"4"}
err = 0
ok = 0
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
    niis = [file for file in files if ("nii.gz" in file and any(center in file for center in centers) and not "uncorrect" in file)]
    if len(niis) < 4:
        print("error not 5 files in folder", subfolder)
        err += 1
    else:
        if folder_name in subfolders2:
            for nii in niis:
                nii_center = (nii.split("_")[-1]).split(".")[0]
                #shutil.copyfile(os.path.join(path,subfolder,"t2s",nii),os.path.join(path2,"derivatives",folder_name,"anat", folder_name + "_T2star" + "_rater" + centers[nii_center]))
                print(os.path.join(path,subfolder,"t2s",nii),os.path.join(path2,"derivatives",folder_name,"anat", folder_name + "_T2star" + "_seg-lesion" + centers[nii_center] + ".nii.gz"))
        ok += 1
    #print(niis)
    #for nii in niis:

    #print(sub_no)
    print(folder_name)
    #print(err,ok)

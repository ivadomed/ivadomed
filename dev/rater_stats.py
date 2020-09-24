import os
import random
import shutil
import nibabel
import numpy as np
import pandas as pd
from skimage import measure

def subjectFilter(input):
    if("sub" in input):
        return True
    else:
        return False

#contrasts = ["FLAIR", "ce-T1w", "PD", "T1w", "T2w"]
contrasts = ["T2w"]
deriv_path = "/scratch/ms_brain/_BIDS_sameResolution/derivatives/labels"
#deriv_path = "/scratch/ms_brain/_BIDS/derivatives/labels"
subjects=list(filter(subjectFilter,os.listdir(deriv_path)))
print(subjects)

df = pd.DataFrame()
df2 = pd.DataFrame()
for subject in subjects:
    files = os.listdir(os.path.join(deriv_path,subject,"anat"))
    niis = [file for file in files if any(contrast in file for contrast in contrasts)]
    dict = {}
    for nii in niis:
        base_name = "_".join((nii.split("_"))[0:2])
        rater = ((nii.split("_")[-1]).split(".")[0])[-1]
        if rater.isnumeric():

        #if rater == "n":
            fname = os.path.join(deriv_path,subject,"anat",nii)
            im1 = nibabel.load(fname).get_data()
            im1[im1 >= 0.5] = 1
            im1[im1 < 0.5] = 0
            dict[rater] = (base_name,im1)
            labels = measure.label(im1)
            df = df.append({'file': base_name, 'rater': rater, 'lesion_count': labels.max(), 'positive_voxels': np.count_nonzero(im1)}, ignore_index=True)
            print(base_name)
            print(rater)

    gt = dict["0"]
    for key in dict:
        if key != "0"
            im1 = (dict[key])[1]
            #Threshold since some files have 3 values [0, 0.2, 1]
            TP = np.logical_and(im1, gt)
            FP = np.logical_and(im1, np.logical_not(gt))
            FN = np.logical_and(np.logical_not(im1), gt)
            TN = np.logical_and(np.logical_not(im1), np.logical_not(gt))
            df2 = df2.append({'file': (dict[key])[0], 'rater': key, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}, ignore_index=True)

print(df.head(30))
df.to_csv('rater_lesion_stats.csv')
df.to_csv('rater_voxel_stats.csv')

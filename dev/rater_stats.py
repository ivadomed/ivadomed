import os
import random
import shutil
import nibabel
import numpy as np
import pandas as pd
from skimage import measure
from ivadomed import transforms as imed_transforms
from scipy.ndimage import zoom

def subjectFilter(input):
    if("sub" in input):
        return True
    else:
        return False



#contrasts = ["FLAIR", "ce-T1w", "PD", "T1w", "T2w"]
contrasts = ["T2star"]
#ms brain
#deriv_path = "/scratch/ms_brain/_BIDS_sameResolution/derivatives/labels"
#deriv_path = "/scratch/ms_brain/_BIDS/derivatives/labels"
#gm
deriv_path = "../duke/projects/ivadomed/gm_challenge_16_inter_rater/derivatives/labels"

subjects=list(filter(subjectFilter,os.listdir(deriv_path)))
print(subjects)

def ms_brain_center_consensus(dict)
    center_1 = dict["1"][2] + dict["2"][2] + dict["4"][2] + dict["5"][2]
    center_1 = np.where(center_1 >= threshold, 1, 0)
    center_2 = dict["3"][2] + dict["6"][2]
    center_2 = np.where(center_2 >= threshold, 1, 0)
    center_3 = dict["7"][2]
    nibabel.save( nibabel.Nifti1Image(center_1, nibabel.load(dict["1"][0]).get_affine()), "_".join(fname.split("_")[0:-1]) + "majority-center1" + ".nii.gz")
    nibabel.save( nibabel.Nifti1Image(center_2, nibabel.load(dict["3"][0]).get_affine()), "_".join(fname.split("_")[0:-1]) + "_majority-center2" + ".nii.gz")
    nibabel.save( nibabel.Nifti1Image(center_3, nibabel.load(dict["7"][0]).get_affine()), "_".join(fname.split("_")[0:-1]) + "_majority-center3" + ".nii.gz")


df = pd.DataFrame()
df2 = pd.DataFrame()
for subject in subjects:
    files = os.listdir(os.path.join(deriv_path,subject,"anat"))
    niis = [file for file in files if any(contrast in file for contrast in contrasts)]
    dict = {}
    print(subject)
    for nii in niis:
        base_name = "_".join((nii.split("_"))[0:2])
        rater = ((nii.split("_")[-1]).split(".")[0])[-1]
        if rater.isnumeric():

            #if rater == "n":

            #If we want to use majority instead of staples for MS brain
            #if rater != 0:
                fname = os.path.join(deriv_path,subject,"anat",nii)
                im1 = nibabel.load(fname).get_data()
                zooms = nibabel.load(fname).header.get_zooms()
                #print(zooms)
                im1[im1 > 0] = 1
                #im1[im1 < 0.5] = 0
                dict[rater] = (fname, base_name, im1, zooms)
                labels = measure.label(im1)
                df = df.append({'path': fname, 'file': base_name, 'rater': rater, 'lesion_count': labels.max(), 'positive_voxels': np.count_nonzero(im1)}, ignore_index=True)
                #print(base_name)
                #print(rater)
    #print(dict)

    ms_brain_center_consensus(dict)

    #Majority voting for gm
    sum = np.zeros((dict["1"][2]).shape)
    for key in dict:
        sum += dict[key][2]
    threshold = 2
    im1 = np.where(sum >= threshold, 1, 0)
    dict["0"] = (None, None, im1, None)
    labels = measure.label(im1)
    df = df.append({'file': "", 'rater': "0", 'lesion_count': labels.max(), 'positive_voxels': np.count_nonzero(im1)}, ignore_index=True)

    gt = (dict["0"])[2]
    for key in dict:
        if key != "0":
            im1 = (dict[key])[2]
            #Threshold since some files have 3 values [0, 0.2, 1]
            TP = np.count_nonzero(np.logical_and(im1, gt))
            FP = np.count_nonzero(np.logical_and(im1, np.logical_not(gt)))
            FN = np.count_nonzero(np.logical_and(np.logical_not(im1), gt))
            TN = np.count_nonzero(np.logical_and(np.logical_not(im1), np.logical_not(gt)))
            total = np.size(gt)
            df2 = df2.append({'file': (dict[key])[1], 'rater': key, 'TP': TP/total, 'FP': FP/total, 'FN': FN/total, 'TN': TN/total}, ignore_index=True)

print(df.head(30))
df.to_csv('rater_lesion_stats.csv')
df2.to_csv('rater_voxel_stats.csv')

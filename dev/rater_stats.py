import os
import random
import shutil
import nibabel
import numpy as np
import pandas as pd

def subjectFilter(input):
    if("sub" in input):
        return True
    else:
        return False

#suffix = "seg-random.nii.gz"
contrasts = ["FLAIR", "ce-T1w", "PD", "T1w", "T2w"]
deriv_path = "/scratch/ms_brain/_BIDS_sameResolution/derivatives/labels"
subjects=list(filter(subjectFilter,os.listdir(deriv_path)))
print(subjects)

df = pd.DataFrame(columns = ['file' , 'rater', 'metric' , 'value'])

for subject in subjects:
    files = os.listdir(os.path.join(deriv_path,subject,"anat"))
    niis = [file for file in files if any(contrast in file for contrast in contrasts)]
        for nii in niis:
            base_name = "_".join((nii.split("_"))[0:2])
            rater = ((nii.split("_")[-1]).split(".")[0])[-1]
            if rater.isnumeric():
                fname = os.path.join(deriv_path,subject,"anat",nii)
                #im1 = nibabel.load(fname).get_data()
                df.append([base_name, rater, "", 0])
                print(base_name)
                print(rater)

print(df.head())


        #new_name = "_".join([name, suffix])
        #print(new_name)
        #print(os.path.join(deriv_path,subject,"anat",new_name))
        #shutil.copyfile(os.path.join(deriv_path,subject,"anat",selected),os.path.join(deriv_path,subject,"anat",new_name))

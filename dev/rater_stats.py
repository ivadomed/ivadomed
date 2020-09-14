import os
import random
import shutil

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
for subject in subjects:
    files = os.listdir(os.path.join(deriv_path,subject,"anat"))
    for contrast in contrasts:
        niis = [file for file in files if (contrast in file)])
        for nii in niis:
            im1 = nibabel.load(os.path.join(deriv_path,subject,nii)).get_data()
        name = "_".join((selected.split("_"))[0:2])
        rater = (selected.split("_"))[-1]
        print(rater)
        #new_name = "_".join([name, suffix])
        #print(new_name)
        #print(os.path.join(deriv_path,subject,"anat",new_name))
        #shutil.copyfile(os.path.join(deriv_path,subject,"anat",selected),os.path.join(deriv_path,subject,"anat",new_name))

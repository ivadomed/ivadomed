import os
import random
import shutil

def subjectFilter(input):
    if("sub" in input):
        return True
    else:
        return False

suffix = "seg-random.nii.gz"
contrasts = ["FLAIR", "ce-T1w", "PD", "T1w", "T2w"]
files = os.listdir("/scratch/ms_brain/_BIDS_sameResolution/derivatives/labels")
subjects=list(filter(subjectFilter,files))
print(subjects)
for subject in subjects:
    niis = os.listdir(os.path.join("/scratch/ms_brain/_BIDS_sameResolution/derivatives/labels",subject,"anat"))
    for contrast in contrasts:
        selected = random.choice([nii for nii in niis if (contrast in nii)])
        name = "_".join((selected.split("_"))[0:2])
        new_name = "_".join([name, suffix])
        print(new_name)
        print(os.path.join("/scratch/ms_brain/_BIDS_sameResolution/derivatives/labels",subject,"anat",new_name))
        shutil.copyfile(os.path.join("/scratch/ms_brain/_BIDS_sameResolution/derivatives/labels",subject,"anat",selected),os.path.join("/scratch/ms_brain/_BIDS_sameResolution/derivatives/labels",subject,"anat",new_name))

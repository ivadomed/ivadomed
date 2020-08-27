import os
import random
import shutil

def subjectFilter(input):
    if("sub" in input):
        return True
    else:
        return False

def subjectFilter(input):
    contrasts = ["FLAIR", "ce-T1w", "PD", "T1w", "T2w"]
    if("sub" in input):
        return True
    else:
        return False

files = os.listdir("/scratch/ms_brain/_BIDS_sameResolution/")
subjects=list(filter(subjectFilter,files))
print(subjects)
for subject in subjects:
    niis = os.listdir("/scratch/ms_brain/_BIDS_sameResolution/")
    for contrast in contrasts:
        selected = random.choice([nii for nii in niis if (contrast in nii)])
        print(selected)
#os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
#shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
#os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")

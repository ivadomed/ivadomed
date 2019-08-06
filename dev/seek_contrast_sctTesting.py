# Usage:
#	python dev/seek_contrast_sctTesting.py
# Example:
#	python dev/seek_contrast_sctTesting.py

import os
from tqdm import tqdm

PATH_SCTTESTING = os.path.join(os.path.expanduser('~'), 'duke', 'sct_testing', 'large')

def run_main():
    if not os.path.isdir(PATH_SCTTESTING):
        print('\nThis folder does not exist: {}'.format(PATH_SCTTESTING))
        print('Please change the path at the top of this file')

    subj_lst = [os.path.join(PATH_SCTTESTING, s, 'anat') for s in os.listdir(PATH_SCTTESTING) if os.path.isdir(os.path.join(PATH_SCTTESTING, s, 'anat'))]
    print('\n{} subjects found.\n'.format(str(len(subj_lst))))

    contrast_lst_lst = []
    for subj_fold in tqdm(subj_lst, desc="Scanning dataset"):
        img_lst = [i for i in os.listdir(subj_fold) if i.endswith('.nii.gz')]
        contrast_cur_lst = ['_'.join(c.split('.nii.gz')[0].split('_')[1:]) for c in img_lst]
        contrast_lst_lst.append(contrast_cur_lst)

    contrast_lst = [sublst for lst in contrast_lst_lst for sublst in lst]
    contrast_lst_noDuplicate = list(set(contrast_lst))
    print('\n{} contrasts found.\n'.format(str(len(contrast_lst_noDuplicate))))

    print("['{}']".format("', '".join(contrast_lst_noDuplicate)))

if __name__ == "__main__":
    run_main()

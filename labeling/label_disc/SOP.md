this script is made for manual labeling of sct_testing/large.
To run a script:
First give it permission: 
chmod 777 SCRIPT_NAME.sh
Run it with ./SCRIPT_NAME.sh TXT_FILE_WITH_SUB_NAME PATH_TO_DUKE SUFFIX_OF_FILE SUFFIX_OF_LABEL_FILE OUTPUT_FOLDER AUTHOR_NAME

- The txt file contain the list of subject you want to process which might not contain all the subject in sct testing large
- Use the absolut path as this is easier
- The suffix of the file you want to process (e.g. _T2w) might differ depending on subject. we recommend making a list according to suffix. 
- The suffix of the label is the same as the previous one so it will open them with the -ilabel option from sct_label_utils 
- in the output folder given by the 5th argument, you will find in BIDS convention sub-xxx/anat/sub-xxx_SUFFIX_OF_LABEL.nii.gz and sub-xxx/anat/sub-xxxSUFFIX_OF_LABEL.json
- All the subject treated will be saved in a file called list_done.txt that will allow you to keep track of progression and subject in output

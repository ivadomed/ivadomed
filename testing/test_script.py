import os
import sys
import subprocess
from csv import writer
from csv import reader
import logging 


def test_script():
    # testing convert to onnx
    subprocess.check_output("ivadomed_convert_to_onnx -m testing_data/model_unet_test.pt -d 2", shell=True)
    
    #testing prepare_dataset_vertebral_labeling
    subprocess.check_output("ivadomed_prepare_dataset_vertebral_labeling -p testing_data/ -s _T2w -a 3",shell=True)
    
    # testing visualize_transform
    command = "ivadomed_visualize_transforms -i testing_data/sub-unf01/anat/sub-unf01_T1w.nii.gz -n " +\
              "2 -c testing_data/model_config.json " +\
              "-r testing_data/derivatives/labels/sub-test001/anat/sub-unf01_T1w_seg-manual.nii.gz -o visuzalize_test"
    subprocess.check_output(command,shell=True)

    # testing extract_small_dataset
    subprocess.check_output("ivadomed_extract_small_dataset -i testing_data/ -o small_dataset/test_script/ -n 1 -c T2w,T1w -d 1",shell=True)

    # testing compare_model
    command = "ivadomed_compare_models -df temporary_results.csv -n 2"
    subprocess.check_output(command,shell=True)


def test_training():
    # Add new file as needed (no empty test/validation)
    # create empty directory for our new files
    os.makedirs("testing_data/sub-test002/anat/", exist_ok=True)
    os.makedirs("testing_data/sub-test003/anat/", exist_ok=True)
    os.makedirs("testing_data/derivatives/labels/sub-test002/anat/", exist_ok=True)
    os.makedirs("testing_data/derivatives/labels/sub-test003/anat/", exist_ok=True)
    os.makedirs("testing_script", exist_ok=True)

    # sub-test002 and sub-test003 will just be copy of our only real testing subject
    command = "cp testing_data/sub-unf01/anat/sub-unf01_T2w.nii.gz testing_data/sub-test002/anat/sub-test002" + \
              "_T2w.nii.gz"
    subprocess.check_output(command, shell=True)

    command = "cp testing_data/sub-unf01/anat/sub-unf01_T2w.nii.gz testing_data/sub-test003/anat/sub-test003" + \
              "_T2w.nii.gz"
    subprocess.check_output(command, shell=True)

    derivatives = "testing_data/derivatives/labels/"
    command = "cp " + derivatives + "sub-unf01/anat/sub-unf01_T2w_seg-manual.nii.gz " + \
              derivatives + "sub-test002/anat/sub-test002" + \
              "_T2w_seg-manual.nii.gz"
    subprocess.check_output(command,shell=True)

    command = "cp " + derivatives + "sub-unf01/anat/sub-unf01_T2w_seg-manual.nii.gz " + \
              derivatives + "sub-test003/anat/sub-test003" + \
              "_T2w_seg-manual.nii.gz"
    subprocess.check_output(command, shell=True)

    command = "cp testing_data/model_unet_test.pt testing_script/best_model.pt"
    subprocess.check_output(command, shell=True)


    list1 = ["sub-test002"]
    list2 = ["sub-test003"]

    # add subjects to participants.tsv
    append_list_as_row("testing_data/participants.tsv", list1)
    append_list_as_row("testing_data/participants.tsv", list2)

    subprocess.check_output(["ivadomed -c testing_data/model_config_test.json"], shell=True)
    subprocess.check_output(["ivadomed -c testing_data/model_config.json"],shell=True)
    


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

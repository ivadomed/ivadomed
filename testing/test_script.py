import os
import sys
import subprocess
from csv import writer
from csv import reader
import logging 


def test_script():
    subprocess.check_output("ivadomed_convert_to_onnx -m testing_data/model_unet_test.pt -d 2", shell=True)

    subprocess.check_output("ivadomed_prepare_dataset_vertebral_labeling -p testing_data/ -s _T2w -a 3",shell=True)

    command = "ivadomed_visualize_transforms -i testing_data/sub-test001/anat/sub-test001_T1w.nii.gz -n " +\
              "2 -c testing_data/model_config.json " +\
              "-r testing_data/derivatives/labels/sub-test001/anat/sub-test001_T1w_seg-manual.nii.gz -o visuzalize_test"
    subprocess.check_output(command,shell=True)


    subprocess.check_output("ivadomed_extract_small_dataset -i testing_data/ -o small_dataset/test_script/ -n 1 -c T2w,T1w -d 1",shell=True)

def test_training():
    log = logging.getLogger
    # Add new file as needed (no empty test/validation)
    os.makedirs("testing_data/sub-test002/anat/", exist_ok=True)
    os.makedirs("testing_data/sub-test003/anat/", exist_ok=True)
    os.makedirs("testing_data/derivatives/labels/sub-test002/anat/", exist_ok=True)
    os.makedirs("testing_data/derivatives/labels/sub-test003/anat/", exist_ok=True)

    command = "cp testing_data/sub-test001/anat/sub-test001_T2w_mid.nii.gz testing_data/sub-test002/anat/sub-test002" + \
              "_T2w_mid.nii.gz"
    subprocess.check_output(command, shell=True)

    command = "cp testing_data/sub-test001/anat/sub-test001_T2w_mid.nii.gz testing_data/sub-test003/anat/sub-test003" + \
              "_T2w_mid.nii.gz"
    subprocess.check_output(command, shell=True)

    derivatives = "testing_data/derivatives/labels/"
    command = "cp " + derivatives + "sub-test001/anat/sub-test001_T2w_mid_heatmap3.nii.gz " + \
              derivatives + "sub-test002/anat/sub-test002" + \
              "_T2w_mid_heatmap3.nii.gz"
    subprocess.check_output(command,shell=True)

    command = "cp " + derivatives + "sub-test001/anat/sub-test001_T2w_mid_heatmap3.nii.gz " + \
              derivatives + "sub-test003/anat/sub-test003" + \
              "_T2w_mid_heatmap3.nii.gz"
    subprocess.check_output(command, shell=True)

    list1 = ["sub-test002"]
    list2 = ["sub-test003"]

    append_list_as_row("testing_data/participants.tsv", list1)
    append_list_as_row("testing_data/participants.tsv", list2)

    #log.debug("training about to begin")
    a = subprocess.call(["ivadomed", "testing_data/model_config_test.json"])
    print(a)
    subprocess.check_output(["ivadomed", "testing_data/model_config.json"],shell=True)
    print("training_done")

    command = "ivadomed_automate_training -c testing_data/model_config.json -p hyperparameter_opt.json -n 1 "
    


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

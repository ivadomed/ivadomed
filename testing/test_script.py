import os


def test_script():
    # ivadomed_convert_to_onnx
    os.system("ivadomed_convert_to_onnx -m testing_data/model_unet_test.pt -d 2")

    # ivadomed_prepare_dataset_vertebral_labeling
    os.system("ivadomed_prepare_dataset_vertebral_labeling -p testing_data/ -s _T2w -a 3")

    # ivadomed_visualize_transforms
    command = "ivadomed_visualize_transforms -i testing_data/sub-test001/anat/sub-test001_T1w.nii.gz -n " +\
              "2 -c testing_data/model_config.json " +\
              "-r testing_data/derivatives/labels/sub-test001/anat/sub-test001_T1w_seg-manual.nii.gz -o ./"
    os.system(command)

    # ivadomed_extract_small_dataset
    os.system("ivadomed_extract_small_dataset -i testing_data/ -o small_dataset/test_script/ -n 1 -c T2w,T1w -d 0")

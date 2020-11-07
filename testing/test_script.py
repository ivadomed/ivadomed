import json
import os
import subprocess
from csv import writer
import shutil

import nibabel as nib
import pytest
import torch

from ivadomed import config_manager as imed_config_manager
import ivadomed.models as imed_models
from ivadomed import main as imed
from ivadomed import utils as imed_utils
from ivadomed import inference as imed_inference


def test_download_data():
    command = "ivadomed_download_data -d t2_tumor"
    subprocess.check_output(command, shell=True)
    command = "ivadomed_download_data -d data_testing -o t2_tumor -k 1"
    subprocess.check_output(command, shell=True)


def test_onnx_conversion():
    # testing convert to onnx
    subprocess.check_output("ivadomed_convert_to_onnx -m testing_data/model_unet_test.pt -d 2", shell=True)


def test_prepare_dataset_vertebral_labeling():
    # testing prepare_dataset_vertebral_labeling
    subprocess.check_output("ivadomed_prepare_dataset_vertebral_labeling -p testing_data/ -s _T2w -a 3", shell=True)


def test_visualize_transform():
    # testing visualize_transform
    command = "ivadomed_visualize_transforms -i testing_data/sub-unf01/anat/sub-unf01_T1w.nii.gz -n " + \
              "2 -c testing_data/model_config.json " + \
              "-r testing_data/derivatives/labels/sub-test001/anat/sub-unf01_T1w_seg-manual.nii.gz -o visuzalize_test"
    subprocess.check_output(command, shell=True)


def test_extract_small():
    # testing extract_small_dataset
    subprocess.check_output("ivadomed_extract_small_dataset -i testing_data/ -o small_dataset/test_script/ -n 1 -c T2w,"
                            "T1w -d 1", shell=True)


def test_compare_model():
    # testing compare_model
    command = "ivadomed_compare_models -df testing_data/temporary_results.csv -n 2 -o output_test.csv"
    subprocess.check_output(command, shell=True)


def test_creation_dataset():
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

    # populate derivatives for sub-test002
    derivatives = "testing_data/derivatives/labels/"
    command = "cp " + derivatives + "sub-unf01/anat/sub-unf01_T2w_seg-manual.nii.gz " + \
              derivatives + "sub-test002/anat/sub-test002" + \
              "_T2w_seg-manual.nii.gz"
    subprocess.check_output(command, shell=True)

    command = "cp " + derivatives + "sub-unf01/anat/sub-unf01_T2w_lesion-manual.nii.gz " + \
              derivatives + "sub-test002/anat/sub-test002" + \
              "_T2w_lesion-manual.nii.gz"
    subprocess.check_output(command, shell=True)

    # populate derivatives for sub-test003
    command = "cp " + derivatives + "sub-unf01/anat/sub-unf01_T2w_seg-manual.nii.gz " + \
              derivatives + "sub-test003/anat/sub-test003" + \
              "_T2w_seg-manual.nii.gz"
    subprocess.check_output(command, shell=True)

    command = "cp " + derivatives + "sub-unf01/anat/sub-unf01_T2w_lesion-manual.nii.gz " + \
              derivatives + "sub-test003/anat/sub-test003" + \
              "_T2w_lesion-manual.nii.gz"
    subprocess.check_output(command, shell=True)

    # Model needs to be inside the log_directory since we use a config file.
    command = "cp testing_data/model_unet_test.pt testing_script/best_model.pt"
    subprocess.check_output(command, shell=True)

    list1 = ["sub-test002"]
    list2 = ["sub-test003"]

    # add subjects to participants.tsv
    append_list_as_row("testing_data/participants.tsv", list1)
    append_list_as_row("testing_data/participants.tsv", list2)


def test_testing_with_uncertainty():
    # Test config. Uses Uncertainty
    subprocess.check_output(["ivadomed -c testing_data/model_config_test.json"], shell=True)


def test_training():
    # Train config
    subprocess.check_output(["ivadomed -c testing_data/model_config.json"], shell=True)


# def test_training_curve():
# using the results from previous training
#   command = "ivadomed_training_curve -i testing_script/ -o training"
#  subprocess.check_output(command, shell=True)


def test_create_eval_json():
    # modify train config
    command = "cp testing_data/model_config.json testing_data/model_config_eval.json"
    subprocess.check_output(command, shell=True)
    file_conf = open("testing_data/model_config_eval.json", "r")
    initial_config = json.load(file_conf)
    file_conf.close()
    file_conf = open("testing_data/model_config_eval.json", "w")
    initial_config["command"] = "test"
    initial_config["transformation"] = {
        "Resample": {
            "wspace": 0.75,
            "hspace": 0.75
        },
        "RandomAffine": {
            "degrees": 4.6,
            "translate": [0.03, 0.03],
            "scale": [0.98, 1]
        },
        "NumpyToTensor": {},
        "NormalizeInstance": {
            "applied_to": ["im"]
        }}
    initial_config["uncertainty"] = {
            "epistemic": True,
            "aleatoric": False,
            "n_it": 2
        }
    initial_config["postprocessing"] = {
            "remove_noise": {"thr": 0.01},
            "keep_largest": {},
            "binarize_prediction": {"thr": 0.5},
            "uncertainty": {"thr": 0.4, "suffix": "_unc-vox.nii.gz"},
            "fill_holes": {},
            "remove_small": {"unit": "vox", "thr": 3}
        }
    json.dump(initial_config, file_conf)


def test_eval():
    subprocess.check_output(["ivadomed -c testing_data/model_config_eval.json"], shell=True)


def test_create_automate_training_json():
    # modify train config
    command = "cp testing_data/model_config_eval.json testing_data/model_config_auto.json"
    subprocess.check_output(command, shell=True)
    file_conf = open("testing_data/model_config_auto.json", "r")
    initial_config = json.load(file_conf)
    file_conf.close()
    file_conf = open("testing_data/model_config_auto.json", "w")
    initial_config["command"] = "train"
    initial_config["gpu"] = "[7]"
    json.dump(initial_config, file_conf)


def test_automate_training_train():
    command = "ivadomed_automate_training -c testing_data/model_config_auto.json " \
              "-p testing_data/hyperparameter_opt.json -n 1 --fixed-split"
    subprocess.check_output(command, shell=True)


def test_training_curve_multiple():
    subprocess.check_output(["ivadomed_training_curve -i ./testing_script-batch_size= --multiple "
                             "-o visu_test_multiple"], shell=True)


def test_automate_training_test():
    command = "ivadomed_automate_training -c testing_data/model_config_auto.json " \
              "-p testing_data/hyperparameter_opt.json -n 1 --run-test --all-combin -t 0.1"
    subprocess.check_output(command, shell=True)


def test_create_json_3d_unet_test():
    # modify train config
    null = None
    command = "cp ivadomed/config/config_tumorSeg.json testing_data/model_config_3d.json"
    subprocess.check_output(command, shell=True)
    file_conf = open("testing_data/model_config_3d.json", "r")
    initial_config = json.load(file_conf)
    file_conf.close()
    file_conf = open("testing_data/model_config_3d.json", "w")
    initial_config["command"] = "test"
    initial_config["loader_parameters"] = {
        "target_suffix": ["_lesion-manual"],
        "roi_suffix": null,
        "bids_path": "testing_data",
        "roi_params": {
            "suffix": null,
            "slice_filter_roi": null
        },
        "contrast_params": {
            "training_validation": ["T2w"],
            "testing": ["T2w"],
            "balance": {}
        },
        "slice_filter_params": {
            "filter_empty_mask": False,
            "filter_empty_input": True
        },
        "slice_axis": "sagittal",
        "multichannel": False,
        "soft_gt": False
    }
    initial_config["log_directory"] = "3d_test"
    initial_config["Modified3DUNet"] = {
        "applied": True,
        "length_3D": [48, 48, 16],
        "stride_3D": [48, 48, 16],
        "attention": True,
        "n_filters": 8
    }
    initial_config["transformation"] = {
        "Resample":
            {
                "wspace": 1,
                "hspace": 1,
                "dspace": 1
            },
        "CenterCrop": {"size": [48, 48, 16]},
        "NumpyToTensor": {},
        "NormalizeInstance": {"applied_to": ["im"]}
    }
    json.dump(initial_config, file_conf)


def test_create_model_unet3d():
    model = imed_models.Modified3DUNet(in_channel=1, out_channel=1, n_filters=8, attention=True)
    torch.save(model, "model_unet_3d_test.pt")
    os.makedirs("3d_test")
    command = "cp model_unet_3d_test.pt 3d_test/best_model.pt"
    subprocess.check_output(command, shell=True)


def test_testing_unet3d():
    subprocess.check_output(["ivadomed -c testing_data/model_config_3d.json"], shell=True)


def test_training_curve_single():
    subprocess.check_output(["ivadomed_training_curve -i testing_script -o visu_test"], shell=True)


@pytest.mark.parametrize('train_lst', [['sub-unf01', 'sub-unf02', 'sub-unf03']])
@pytest.mark.parametrize('target_lst', [["_lesion-manual"]])
@pytest.mark.parametrize('config', [
    {
        "object_detection_params": {
            "object_detection_path": "findcord_tumor",
            "safety_factor": None,
            "log_directory": "testing_script"
        },
        "transformation": {
            "Resample": {
                "wspace": 0.75,
                "hspace": 0.75,
                "dspace": 0.75
            },
            "CenterCrop": {
                "size": [32, 32, 32]
            },
            "NumpyToTensor": {}
        },
        "Modified3DUNet": {
            "applied": True,
            "length_3D": [32, 32, 32],
            "stride_3D": [32, 32, 32],
            "attention": False,
            "n_filters": 8
        },
        "split_dataset": {
            "fname_split": None,
            "random_seed": 1313,
            "method": "per_patient",
            "train_fraction": 0.34,
            "test_fraction": 0.33,
            "center_test": []
        },
    }])
def test_object_detection(train_lst, target_lst, config):
    # Load config file
    context = imed_config_manager.ConfigurationManager("testing_data/model_config.json").get_config()
    context.update(config)

    command = "ivadomed_download_data -d findcord_tumor"
    subprocess.check_output(command, shell=True)

    imed.run_command(context)


def test_object_detection_inference():
    fname_image = "testing_data/sub-unf01/anat/sub-unf01_T1w.nii.gz"

    # Detection
    nib_detection = imed_inference.segment_volume(folder_model="findcord_tumor", fname_image=fname_image)
    detection_file = "detection.nii.gz"
    nib.save(nib_detection, detection_file)

    # Segmentation
    imed_inference.segment_volume(folder_model="t2_tumor", fname_image=fname_image, options={'fname_prior': detection_file})


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def test_film_contrast():
    # FiLM config
    # Create config copy
    base_config = "testing_data/model_config.json"
    film_config = "testing_data/model_config_film.json"
    subprocess.check_output(" ".join(["cp", base_config, film_config]), shell=True)
    # Read config
    with open(film_config, "r") as fhandle:
        context = json.load(fhandle)
    # Modify params
    context["loader_parameters"]["contrast_params"]["training_validation"] = ["T2w", "T1w", "T2star"]
    context["FiLMedUnet"]["applied"] = True
    context["FiLMedUnet"]["film_layers"] = 2 * [1] * context["default_model"]["depth"] + [0, 0]
    context["debugging"] = True
    # Save modified config file
    with open(film_config, 'w') as fp:
        json.dump(context, fp, indent=4)

    # Run command
    command = "ivadomed -c " + film_config
    subprocess.check_output(command, shell=True)


def test_resume_training():
    # Train config
    subprocess.check_output(["ivadomed -c testing_data/model_config.json"], shell=True)

    # Add few epochs copy
    base_config = "testing_data/model_config.json"
    with open(base_config, "r") as fhandle:
        context = json.load(fhandle)
    context["training_parameters"]["training_time"]["num_epochs"] += 2
    with open(base_config, 'w') as fp:
        json.dump(context, fp, indent=4)

    # Resume training
    subprocess.check_output(["ivadomed -c testing_data/model_config.json --resume"], shell=True)

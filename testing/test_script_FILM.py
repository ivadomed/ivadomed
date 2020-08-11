import os
import sys
import subprocess
from csv import writer
from csv import reader
import logging
import json
import ivadomed.models as imed_models
import torch


def copy_metadata():
    command = "cp testing_data/sub-unf01/anat/sub-unf01_T2w.json testing_data/sub-test002/anat/sub-test002" + \
              "_T2w.json"
    subprocess.check_output(command, shell=True)

    command = "cp testing_data/sub-unf01/anat/sub-unf01_T2w.json testing_data/sub-test003/anat/sub-test003" + \
              "_T2w.json"
    subprocess.check_output(command, shell=True)


def test_create_json_film():
    # modify train config
    null = None
    command = "cp ivadomed/config/config.json testing_data/model_config_film.json"
    subprocess.check_output(command, shell=True)
    file_conf = open("testing_data/model_config_film.json", "r")
    initial_config = json.load(file_conf)
    file_conf.close()
    file_conf = open("testing_data/model_config_film.json", "w")
    initial_config["loader_parameters"] = {
        "target_suffix": ["_seg-manual"],
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

    initial_config["training_parameters"] = {
        "batch_size": 18,
        "loss": {
            "name": "DiceLoss"
        },
        "training_time": {
            "num_epochs": 1,
            "early_stopping_patience": 50,
            "early_stopping_epsilon": 0.001
        },
        "scheduler": {
            "initial_lr": 0.001,
            "lr_scheduler": {
                "name": "CosineAnnealingLR",
                "base_lr": 1e-5,
                "max_lr": 1e-2
            }
        },
        "balance_samples": False,
        "mixup_alpha": null,
        "transfer_learning": {
            "retrain_model": null,
            "retrain_fraction": 1.0
        }
    }

    initial_config["FiLMedUnet"] = {
        "applied": True,
        "metadata": "mri_params",
        "film_layers": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    initial_config["transformation"]={
        "Resample":
        {
            "wspace": 0.75,
            "hspace": 0.75,
            "dspace": 1,
            "preprocessing": True
        },
        "CenterCrop": {
            "size": [64, 64],
            "preprocessing": True
        },
      "NumpyToTensor": {},
      "NormalizeInstance": {"applied_to": ["im"]}
    }
    json.dump(initial_config, file_conf)


def test_film_train():
    subprocess.check_output(["ivadomed -c testing_data/model_config_film.json"], shell=True)

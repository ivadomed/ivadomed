#!/usr/bin/env python
##############################################################
#
# This script is used to test the dataframe of the new loader
# and the new splitting method
# This script was used to generate the df_ref for data-testing
#
# Usage: python dev/df_new_loader.py
#
##############################################################


# IMPORTS
import json
import os
import argparse
import copy
import joblib
import torch.backends.cudnn as cudnn
import nibabel as nib
import sys
import platform
import multiprocessing
import pandas as pd

from bids_neuropoly import bids
from ivadomed.utils import logger
from ivadomed import evaluation as imed_evaluation
from ivadomed import config_manager as imed_config_manager
from ivadomed import testing as imed_testing
from ivadomed import training as imed_training
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed import metrics as imed_metrics
from ivadomed import inference as imed_inference
# Here imed_loader refers temporarily to the loader_new.py, to fix when integrating in pipeline
from ivadomed.loader import utils as imed_loader_utils, loader_new as imed_loader, film as imed_film


### STEP 1 - CREATE BidsDataframe object

# GET LOADER PARAMETERS FROM IVADOMED CONFIG FILE
# The loader parameters have 2 new fields: "bids_config" and "extensions".
# "bids_config" is mandatory for microscopy until BEP is merged and pybids is updated, the file is
# in ivadomed/config/config_bids.json.
# "bids_config" is optional for anat
# "extensions" is used to filter which files are to be indexed, in case multiple file types are present.
path_config_file = "ivadomed/config/config_new_loader.json"
context = imed_config_manager.ConfigurationManager(path_config_file).get_config()
loader_params = context["loader_parameters"]
loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["training_validation"]

# CHOOSE TO INDEX DERIVATIVES OR NOT
# As per pybids, the indexing of derivatives works only if a "dataset_description.json" file
# is present in "derivatives" or "labels" folder with minimal content:
# {"Name": "Example dataset", "BIDSVersion": "1.0.2", "PipelineDescription": {"Name": "Example pipeline"}}
derivatives = True

# CREATE OUTPUT PATH
path_output = context["path_output"]
if not os.path.isdir(path_output):
    print('Creating output path: {}'.format(path_output))
    os.makedirs(path_output)
else:
    print('Output path already exists: {}'.format(path_output))

# CREATE BIDSDataframe OBJECT
bids_df = imed_loader_utils.BidsDataframe(loader_params, derivatives, path_output)
df = bids_df.df

# DROP "path" AND "parent_path" COLUMNS AND SORT BY FILENAME FOR TESTING PURPOSES WITH data-testing
#df = df.drop(columns=['path', 'parent_path'])
#df = df.sort_values(by=['filename']).reset_index(drop=True)

# SAVE DATAFRAME TO CSV FILE FOR data-testing
#path_csv = "test_df_new_loader.csv"
#df.to_csv(path_csv, index=False)
#print(df)

### STEP 2 - SPLIT DATASET

# SPLIT TRAIN/VALID/TEST (with "new" functions)
train_lst, valid_lst, test_lst = imed_loader_utils.get_subdatasets_subjects_list_new(context["split_dataset"],
                                                                                     bids_df.df,
                                                                                     path_output,
                                                                                     context["loader_parameters"]
                                                                                     ['subject_selection'])
#print("Train:", train_lst)
#print("Valid:", valid_lst)
#print("Test:", test_lst)

### STEP 3 - LOAD DATASET (DERIVED FROM MAIN)

# List of not-default available models i.e. different from Unet
MODEL_LIST = ['Modified3DUNet', 'HeMISUnet', 'FiLMedUnet', 'resnet18', 'densenet121', 'Countception']

command = copy.deepcopy(context["command"])
path_output = copy.deepcopy(context["path_output"])
if not os.path.isdir(path_output):
    print('Creating output path: {}'.format(path_output))
    os.makedirs(path_output)
else:
    print('Output path already exists: {}'.format(path_output))

cuda_available, device = imed_utils.define_device(context['gpu_ids'][0])

# Loader params
loader_params = copy.deepcopy(context["loader_parameters"])
if command == "train":
    loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["training_validation"]
else:
    loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["testing"]
if "FiLMedUnet" in context and context["FiLMedUnet"]["applied"]:
    loader_params.update({"metadata_type": context["FiLMedUnet"]["metadata"]})
# Load metadata necessary to balance the loader
if context['training_parameters']['balance_samples']['applied'] and \
        context['training_parameters']['balance_samples']['type'] != 'gt':
    loader_params.update({"metadata_type": context['training_parameters']['balance_samples']['type']})
# Get transforms for each subdataset
transform_train_params, transform_valid_params, transform_test_params = \
    imed_transforms.get_subdatasets_transforms(context["transformation"])
# MODEL PARAMETERS
model_params = copy.deepcopy(context["default_model"])
model_params["folder_name"] = copy.deepcopy(context["model_name"])
model_context_list = [model_name for model_name in MODEL_LIST
                      if model_name in context and context[model_name]["applied"]]
if len(model_context_list) == 1:
    model_params["name"] = model_context_list[0]
    model_params.update(context[model_context_list[0]])
elif 'Modified3DUNet' in model_context_list and 'FiLMedUnet' in model_context_list and len(model_context_list) == 2:
    model_params["name"] = 'Modified3DUNet'
    for i in range(len(model_context_list)):
        model_params.update(context[model_context_list[i]])
elif len(model_context_list) > 1:
    print('ERROR: Several models are selected in the configuration file: {}.'
          'Please select only one (i.e. only one where: "applied": true).'.format(model_context_list))
    exit()
model_params['is_2d'] = False if "Modified3DUNet" in model_params['name'] else model_params['is_2d']
# Get in_channel from contrast_lst
if loader_params["multichannel"]:
    model_params["in_channel"] = len(loader_params["contrast_params"]["contrast_lst"])
else:
    model_params["in_channel"] = 1
# Get out_channel from target_suffix
model_params["out_channel"] = len(loader_params["target_suffix"])
# If multi-class output, then add background class
if model_params["out_channel"] > 1:
    model_params.update({"out_channel": model_params["out_channel"] + 1})
# Display for spec' check
imed_utils.display_selected_model_spec(params=model_params)
# Update loader params
if 'object_detection_params' in context:
    object_detection_params = context['object_detection_params']
    object_detection_params.update({"gpu_ids": context['gpu_ids'][0],
                                    "path_output": context['path_output']})
    loader_params.update({"object_detection_params": object_detection_params})
loader_params.update({"model_params": model_params})
# TESTING PARAMS
# Aleatoric uncertainty
if context['uncertainty']['aleatoric'] and context['uncertainty']['n_it'] > 0:
    transformation_dict = transform_train_params
else:
    transformation_dict = transform_test_params
undo_transforms = imed_transforms.UndoCompose(imed_transforms.Compose(transformation_dict, requires_undo=True))
testing_params = copy.deepcopy(context["training_parameters"])
testing_params.update({'uncertainty': context["uncertainty"]})
testing_params.update({'target_suffix': loader_params["target_suffix"], 'undo_transforms': undo_transforms,
                       'slice_axis': loader_params['slice_axis']})
if command == "train":
    imed_utils.display_selected_transfoms(transform_train_params, dataset_type=["training"])
    imed_utils.display_selected_transfoms(transform_valid_params, dataset_type=["validation"])
elif command == "test":
    imed_utils.display_selected_transfoms(transformation_dict, dataset_type=["testing"])
# Check if multiple raters
if any([isinstance(class_suffix, list) for class_suffix in loader_params["target_suffix"]]):
    print(
        "\nAnnotations from multiple raters will be used during model training, one annotation from one rater "
        "randomly selected at each iteration.\n")
    if command != "train":
        print(
            "\nERROR: Please provide only one annotation per class in 'target_suffix' when not training a model.\n")
        exit()
if command == 'train':
    # LOAD DATASET
    # Get Validation dataset
    ds_valid = imed_loader.load_dataset(bids_df, **{**loader_params,
                                           **{'data_list': valid_lst, 'transforms_params': transform_valid_params,
                                              'dataset_type': 'validation'}}, device=device,
                                        cuda_available=cuda_available)

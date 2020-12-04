#!/usr/bin/env python
##############################################################
#
# This script is used to test the dataframe of the new loader
#
# Usage: python dev/df_new_loader.py
#
##############################################################

# IMPORTS
import os
import pandas as pd
from ivadomed import config_manager as imed_config_manager
from ivadomed.loader import utils as imed_loader_utils

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

# CREATE DATAFRAME
df = imed_loader_utils.create_bids_dataframe(loader_params, derivatives)

# SAVE DATAFRAME TO CSV FILE
path_csv = "test_df_new_loader.csv"
df.to_csv(path_csv)
print(df)

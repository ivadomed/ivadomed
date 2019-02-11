#!/bin/bash
# Environment variables for bids dataset processing
# A completely new bids dataset will be generated, containing the orignal
# datas and the derivatives

# Set every other path relative to this path for convenience
# Do not add "/" at the end. Path should be absolute (i.e. do not use "~")
export PATH_PARENT="/Users/nipin_local/test"

# Path to the folder site which contains all sites.
# Do not add "/" at the end. Path should be absolute (i.e. do not use "~")
export PATH_DATA="${PATH_PARENT}/spineGeneric_multiSubjects"

# List of subjects to analyse. Comment this variable if you want to analyze all
# sites in the PATH_DATA folder.
#export SITES=(
#	"ucl"
#	"unf"
#)

# Paths to where to save the new dataset.
# Do not add "/" at the end. Path should be absolute (i.e. do not use "~")
export PATH_OUTPUT="${PATH_PARENT}/spineGeneric_result"

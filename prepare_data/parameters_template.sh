#!/bin/bash
# Environment variables for bids dataset processing
# A completely new bids dataset will be generated, containing the orignal
# datas and the derivatives

# Set every other path relative to this path for convenience
# Do not add "/" at the end. Path should be absolute (i.e. do not use "~")
export PATH_PARENT="/Users/julien"

# Path to the folder site which contains all sites.
# Do not add "/" at the end. Path should be absolute (i.e. do not use "~")
export PATH_DATA="${PATH_PARENT}/spineGeneric_multiSubjects"

# List of subjects to analyse. Comment this variable if you want to analyze all
# sites in the PATH_DATA folder.
#export SITES=(
#	"amu_spineGeneric"
#)

# Paths to where to save the new dataset.
# Do not add "/" at the end. Path should be absolute (i.e. do not use "~")
export PATH_OUTPUT="${PATH_PARENT}/spineGeneric/result"
export PATH_QC="${PATH_PARENT}/spineGeneric/qc"
export PATH_LOG="${PATH_PARENT}/spineGeneric/log"
export PATH_SEGMANUAL="${PATH_PARENT}/spineGeneric/seg_manual"  # Location of manually-corrected segmentations

# Misc
export JOBS=20  # Number of jobs for parallel processing

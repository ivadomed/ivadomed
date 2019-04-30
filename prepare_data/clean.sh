#!/bin/bash
#
# Clean if files were mistakenely generated in root folder
#
# Usage:
#   ./clean.sh

# Immediately exit if error
set -v

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1
SITE=$2
PATH_OUTPUT=$3
PATH_QC=$4
PATH_LOG=$5

cd ${SUBJECT}/anat
rm straightening.cache
rm sub-01_acq-T1w_MTS_seg.nii.gz
rm sub-01_T1w_seg.nii.gz
rm sub-01_T2star_seg.nii.gz
rm sub-01_T2w_seg.nii.gz

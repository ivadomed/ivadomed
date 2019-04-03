#!/bin/bash
#
# Check the presence of input files.
#
# Usage:
#   ./check_input_files.sh <SUBJECT> <SITE> <PATH_OUTPUT> <PATH_QC> <PATH_LOG>

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1
SITE=$2
PATH_OUTPUT=$3
PATH_QC=$4
PATH_LOG=$5

# Create BIDS architecture
PATH_IN="`pwd`/${SUBJECT}/anat"

# Set filenames
file_t1w_mts="${SUBJECT}_acq-T1w_MTS"
file_mton="${SUBJECT}_acq-MTon_MTS"
file_mtoff="${SUBJECT}_acq-MToff_MTS"
file_t2w="${SUBJECT}_T2w"
file_t2s="${SUBJECT}_T2star"
file_t1w="${SUBJECT}_T1w"

# Verify presence of output files and write log file if error
FILES_TO_CHECK=(
  "$PATH_IN/${file_t1w_mts}.nii.gz"
  "$PATH_IN/${file_mton}.nii.gz"
  "$PATH_IN/${file_mtoff}.nii.gz"
  "$PATH_IN/${file_t2w}.nii.gz"
  "$PATH_IN/${file_t2s}.nii.gz"
  "$PATH_IN/${file_t1w}.nii.gz"
)
for file in ${FILES_TO_CHECK[@]}; do
  if [ ! -e $file ]; then
    echo "${SITE}/${file} does not exist" >> $PATH_LOG/_error_check_input_files.log
  fi
done

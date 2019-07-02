#!/bin/bash
#
# Delete temporary files. Run this script once all files have been verified.
# Usage:
#   ./delete_temp_files.sh <SUBJECT> <PATH_OUTPUT> <PATH_QC> <PATH_LOG>
#
# -x: Full verbose, -e: Exit if error
set -x

# Retrieve input params
SUBJECT=$1
PATH_OUTPUT=$2
PATH_QC=$3
PATH_LOG=$4

# Create BIDS architecture
ofolder_reg="${PATH_OUTPUT}/${SITE}/${SUBJECT}/anat"

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# Remove tmp folder
rm -rf tmp

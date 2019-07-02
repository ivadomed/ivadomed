#!/bin/bash
#
# Delete temporary files. Run this script once all files have been verified.
# Usage:
#   ./delete_temp_files.sh <SUBJECT> <PATH_OUTPUT> <PATH_QC> <PATH_LOG>

# Uncomment for full verbose
set -v

# Immediately exit if error
set -e

# Retrieve input params
SUBJECT=$1
FILEPARAM=$2

source $FILEPARAM

# Create BIDS architecture
ofolder_reg="${PATH_OUTPUT}/${SITE}/${SUBJECT}/anat"

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# Remove tmp folder
rm -rf tmp

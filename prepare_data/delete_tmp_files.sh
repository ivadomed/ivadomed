#!/bin/bash
#
# Delete temporary files. Run this script once all files have been verified.
# Usage:
#   ./delete_temp_files.sh <SUBJECT> <SITE> <PATH_OUTPUT> <PATH_QC> <PATH_LOG>
#
# -x: Full verbose, -e: Exit if error
set -x

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# Remove tmp folder
rm -rf tmp

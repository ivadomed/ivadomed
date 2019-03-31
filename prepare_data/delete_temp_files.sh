#!/bin/bash
#
# Delete temporary files. Run this script once all files have been verified.

# Retrieve input params
sub=$1
PATH_OUTPUT_wSITE=$2
PATH_QC=$3

# Create BIDS architecture
PATH_IN="`pwd`/${sub}/anat"
ofolder_seg="${PATH_OUTPUT_wSITE}/derivatives/labels/${sub}/anat"
ofolder_reg="${PATH_OUTPUT_wSITE}/${sub}/anat"
mkdir -p ${ofolder_reg}
mkdir -p ${ofolder_seg}

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# Delete temporary files (they interfer with the BIDS wrapper)
rm *_mask.nii.gz
rm warp*
rm *_r_reg.*
rm *crop.*
rm *mean.*
rm *T2star_reg.*
rm ${sub}_acq-T1w_MTS.nii.gz
rm ${ofolder_seg}/warp*
rm ${ofolder_seg}/${sub}_T1w_reg_seg.nii.gz
rm ${ofolder_seg}/${sub}_acq-T1w_MTS_crop_seg_reg.nii.gz
rm ${ofolder_seg}/${sub}_T2star_reg_seg.nii.gz
rm ${ofolder_seg}/${sub}_T2w_reg_seg.nii.gz
rm ${ofolder_seg}/${sub}_*reg.*
rm ${ofolder_seg}/${sub}_*seg.*
rm ${ofolder_seg}/tmp.*

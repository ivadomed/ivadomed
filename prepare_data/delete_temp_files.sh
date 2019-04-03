#!/bin/bash
#
# Delete temporary files. Run this script once all files have been verified.

# Retrieve input params
SUBJECT=$1
SITE=$2
PATH_OUTPUT=$3
PATH_QC=$4
PATH_LOG=$5

# Create BIDS architecture
PATH_OUTPUT_wSITE=${PATH_OUTPUT}/${SITE}
PATH_IN="`pwd`/${SUBJECT}/anat"
ofolder_seg="${PATH_OUTPUT_wSITE}/derivatives/labels/${SUBJECT}/anat"
ofolder_reg="${PATH_OUTPUT_wSITE}/${SUBJECT}/anat"

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# Set filenames
file_t1w_mts="${SUBJECT}_acq-T1w_MTS"
file_mton="${SUBJECT}_acq-MTon_MTS"
file_mtoff="${SUBJECT}_acq-MToff_MTS"
file_t2w="${SUBJECT}_T2w"
file_t2s="${SUBJECT}_T2star"
file_t1w="${SUBJECT}_T1w"

# Delete temporary files (they interfer with the BIDS wrapper)
rm *_mask.nii.gz
rm warp*
rm *_r_reg.*
rm *crop.*
rm *mean.*
rm *T2star_reg.*
rm ${SUBJECT}_acq-T1w_MTS.nii.gz
rm ${ofolder_seg}/warp*
rm ${ofolder_seg}/${SUBJECT}_T1w_reg_seg.nii.gz
rm ${ofolder_seg}/${SUBJECT}_acq-T1w_MTS_crop_seg_reg.nii.gz
rm ${ofolder_seg}/${SUBJECT}_T2star_reg_seg.nii.gz
rm ${ofolder_seg}/${SUBJECT}_T2w_reg_seg.nii.gz
rm ${ofolder_seg}/${SUBJECT}_*reg.*
rm ${ofolder_seg}/${SUBJECT}_*seg.*
rm ${ofolder_seg}/tmp.*

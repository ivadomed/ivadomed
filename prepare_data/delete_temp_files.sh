#!/bin/bash
#
# Delete temporary files. Run this script once all files have been verified.
# Usage:
#   ./delete_temp_files.sh <SUBJECT> <SITE> <PATH_OUTPUT> <PATH_QC> <PATH_LOG>

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

# Duplicate segmentation to be used by other contrasts
rsync -avzh ${ofolder_seg}/${file_t1w_mts}_seg-manual.nii.gz ${ofolder_seg}/${file_mtoff}_seg-manual.nii.gz
rsync -avzh ${ofolder_seg}/${file_t1w_mts}_seg-manual.nii.gz ${ofolder_seg}/${file_mton}_seg-manual.nii.gz
rsync -avzh ${ofolder_seg}/${file_t1w_mts}_seg-manual.nii.gz ${ofolder_seg}/${file_t2w}_seg-manual.nii.gz
rsync -avzh ${ofolder_seg}/${file_t1w_mts}_seg-manual.nii.gz ${ofolder_seg}/${file_t2s}_seg-manual.nii.gz
rsync -avzh ${ofolder_seg}/${file_t1w_mts}_seg-manual.nii.gz ${ofolder_seg}/${file_t1w}_seg-manual.nii.gz

# Copy json files and rename them
rsync -avzh ${PATH_IN}/${SUBJECT}_acq-T1w_MTS.json ${file_t1w_mts}.json
rsync -avzh ${PATH_IN}/${SUBJECT}_acq-MTon_MTS.json ${file_mton}.json
rsync -avzh ${PATH_IN}/${SUBJECT}_acq-MToff_MTS.json ${file_mtoff}.json
rsync -avzh ${PATH_IN}/${SUBJECT}_T2w.json ${file_t2w}.json
rsync -avzh ${PATH_IN}/${SUBJECT}_T2star.json ${file_t2s}.json
rsync -avzh ${PATH_IN}/${SUBJECT}_T1w.json ${file_t1w}.json
rsync -avzh ${PATH_IN}/../../dataset_description.json ../../
rsync -avzh ${PATH_IN}/../../participants.json ../../
rsync -avzh ${PATH_IN}/../../participants.tsv ../../

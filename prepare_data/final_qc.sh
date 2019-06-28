#!/bin/bash
#
# Delete temporary files. Run this script once all files have been verified.
# Usage:
#   ./delete_temp_files.sh <SUBJECT> <SITE> <PATH_OUTPUT> <PATH_QC> <PATH_LOG>
#
# -x: Full verbose, -e: Exit if error
set -x

# Retrieve input params
SUBJECT=$1
PATH_OUTPUT=$2
PATH_QC=$3
PATH_LOG=$4

# Create BIDS architecture
PATH_IN="`pwd`/${SUBJECT}/anat"
ofolder_seg="${PATH_OUTPUT}/derivatives/labels/${SUBJECT}/anat"
ofolder_reg="${PATH_OUTPUT}/${SUBJECT}/anat"

# Set filenames
file_t1w_mts="${SUBJECT}_acq-T1w_MTS"
file_mton="${SUBJECT}_acq-MTon_MTS"
file_mtoff="${SUBJECT}_acq-MToff_MTS"
file_t2w="${SUBJECT}_T2w"
file_t2s="${SUBJECT}_T2star"
file_t1w="${SUBJECT}_T1w"

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# Copy files to appropriate locations
cp tmp/${file_t1w_mts}_crop_r.nii.gz ${file_t1w_mts}.nii.gz
cp tmp/${file_mton}_reg.nii.gz ${file_mton}.nii.gz
cp tmp/${file_mtoff}_reg.nii.gz ${file_mtoff}.nii.gz

# Rename current files (remove "_reg")
mv ${file_t1w}_reg.nii.gz ${file_t1w}.nii.gz
mv ${file_t2w}_reg.nii.gz ${file_t2w}.nii.gz
mv ${file_t2s}_reg.nii.gz ${file_t2s}.nii.gz

FILES=(
  "${file_t1w_mts}"
  "${file_mton}"
  "${file_mtoff}"
  "${file_t2w}"
  "${file_t2s}"
  "${file_t1w}"
)
for file in ${FILES[@]}; do
  # Duplicate segmentation to be used by other contrasts
  cp tmp/${file_t1w_mts}_crop_r_seg-manual.nii.gz ${ofolder_seg}/${file}_seg-manual.nii.gz
  # Remove empty slices at the edge
  prepdata -i ${file}.nii.gz -s ${ofolder_seg}/${file}_seg-manual.nii.gz remove-slice
  # Generate final QC
  sct_qc -i ${file}.nii.gz -s ${ofolder_seg}/${file}_seg-manual.nii.gz -p sct_deepseg_sc -qc ${PATH_QC}2 -qc-dataset ${SITE} -qc-subject ${SUBJECT}
done

# Copy json files and rename them
cp ${PATH_IN}/${SUBJECT}_acq-T1w_MTS.json ${file_t1w_mts}.json
cp ${PATH_IN}/${SUBJECT}_acq-MTon_MTS.json ${file_mton}.json
cp ${PATH_IN}/${SUBJECT}_acq-MToff_MTS.json ${file_mtoff}.json
cp ${PATH_IN}/${SUBJECT}_T2w.json ${file_t2w}.json
cp ${PATH_IN}/${SUBJECT}_T2star.json ${file_t2s}.json
cp ${PATH_IN}/${SUBJECT}_T1w.json ${file_t1w}.json
cp ${PATH_IN}/../../dataset_description.json ../../
cp ${PATH_IN}/../../participants.json ../../
cp ${PATH_IN}/../../participants.tsv ../../

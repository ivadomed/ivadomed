#!/bin/bash
#
# Delete temporary files. Run this script once all files have been verified.
# Usage:
#   ./delete_temp_files.sh <SUBJECT> <SITE> <PATH_OUTPUT> <PATH_QC> <PATH_LOG>
#
# For full verbose, uncomment the next line
set -x

# Retrieve input params
SUBJECT=$1
SITE=$2
PATH_OUTPUT=$3
PATH_QC=$4
PATH_LOG=$5

# Create BIDS architecture
PATH_IN="`pwd`/${SUBJECT}/anat"
ofolder_seg="${PATH_OUTPUT}/${SITE}/derivatives/labels/${SUBJECT}/anat"
ofolder_reg="${PATH_OUTPUT}/${SITE}/${SUBJECT}/anat"

# Set filenames
file_t1w_mts="${SUBJECT}_acq-T1w_MTS"
file_mton="${SUBJECT}_acq-MTon_MTS"
file_mtoff="${SUBJECT}_acq-MToff_MTS"
file_t2w="${SUBJECT}_T2w"
file_t2s="${SUBJECT}_T2star"
file_t1w="${SUBJECT}_T1w"

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# Set filenames
file_t1w_mts="${SUBJECT}_acq-T1w_MTS"
file_mton="${SUBJECT}_acq-MTon_MTS"
file_mtoff="${SUBJECT}_acq-MToff_MTS"
file_t2w="${SUBJECT}_T2w"
file_t2s="${SUBJECT}_T2star"
file_t1w="${SUBJECT}_T1w"

# Copy files to appropriate locations
rsync -avzh tmp/${file_t1w_mts}_crop_r.nii.gz ${file_t1w_mts}.nii.gz
rsync -avzh tmp/${file_mton}_reg.nii.gz ${file_mton}.nii.gz
rsync -avzh tmp/${file_mtoff}_reg.nii.gz ${file_mtoff}.nii.gz

# Rename current files (remove "_reg")
mv ${file_t1w}_reg.nii.gz ${file_t1w}.nii.gz
mv ${file_t2w}_reg.nii.gz ${file_t2w}.nii.gz
mv ${file_t2s}_reg.nii.gz ${file_t2s}.nii.gz

# Duplicate segmentation to be used by other contrasts
rsync -avzh tmp/${file_t1w_mts}_crop_r_seg-manual.nii.gz ${ofolder_seg}/${file_t1w_mts}_seg-manual.nii.gz
rsync -avzh tmp/${file_t1w_mts}_crop_r_seg-manual.nii.gz ${ofolder_seg}/${file_mtoff}_seg-manual.nii.gz
rsync -avzh tmp/${file_t1w_mts}_crop_r_seg-manual.nii.gz ${ofolder_seg}/${file_mton}_seg-manual.nii.gz
rsync -avzh tmp/${file_t1w_mts}_crop_r_seg-manual.nii.gz ${ofolder_seg}/${file_t2w}_seg-manual.nii.gz
rsync -avzh tmp/${file_t1w_mts}_crop_r_seg-manual.nii.gz ${ofolder_seg}/${file_t2s}_seg-manual.nii.gz
rsync -avzh tmp/${file_t1w_mts}_crop_r_seg-manual.nii.gz ${ofolder_seg}/${file_t1w}_seg-manual.nii.gz

# Remove empty slices at the edge
FILES=(
  "${file_t1w_mts}"
  "${file_mton}"
  "${file_mtoff}"
  "${file_t2w}"
  "${file_t2s}"
  "${file_t1w}"
)
for file in ${FILES[@]}; do
  prepdata -i ${file}.nii.gz -s ${ofolder_seg}/${file}_seg-manual.nii.gz remove-slice
done

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

# Generate final QC
sct_qc -i ${file_t1w_mts}.nii.gz -s ${ofolder_seg}/${file_t1w_mts}_seg-manual.nii.gz -p sct_deepseg_sc -qc ${PATH_QC}2 -qc-dataset ${SITE} -qc-subject ${SUBJECT}
sct_qc -i ${file_mton}.nii.gz -s ${ofolder_seg}/${file_mton}_seg-manual.nii.gz -p sct_deepseg_sc -qc ${PATH_QC}2 -qc-dataset ${SITE} -qc-subject ${SUBJECT}
sct_qc -i ${file_mtoff}.nii.gz -s ${ofolder_seg}/${file_mtoff}_seg-manual.nii.gz -p sct_deepseg_sc -qc ${PATH_QC}2 -qc-dataset ${SITE} -qc-subject ${SUBJECT}
sct_qc -i ${file_t1w}.nii.gz -s ${ofolder_seg}/${file_t1w}_seg-manual.nii.gz -p sct_deepseg_sc -qc ${PATH_QC}2 -qc-dataset ${SITE} -qc-subject ${SUBJECT}
sct_qc -i ${file_t2w}.nii.gz -s ${ofolder_seg}/${file_t2w}_seg-manual.nii.gz -p sct_deepseg_sc -qc ${PATH_QC}2 -qc-dataset ${SITE} -qc-subject ${SUBJECT}
sct_qc -i ${file_t2s}.nii.gz -s ${ofolder_seg}/${file_t2s}_seg-manual.nii.gz -p sct_deepseg_sc -qc ${PATH_QC}2 -qc-dataset ${SITE} -qc-subject ${SUBJECT}

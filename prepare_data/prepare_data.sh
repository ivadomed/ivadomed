#!/bin/bash
#
# Generate segmentation and co-register all multimodal data.
#
# Usage:
#   ./prepare_data.sh <SUBJECTject_ID> <output path>
#
# Where SUBJECTject_ID refers to the SUBJECTject ID according to the BIDS format.
#
# Example:
#   ./prepare_data.sh SUBJECT-03 /users/jondoe/bids_data_results/site-01
#

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
ofolder_seg="${PATH_OUTPUT}/${SITE}/derivatives/labels/${SUBJECT}/anat"
ofolder_reg="${PATH_OUTPUT}/${SITE}/${SUBJECT}/anat"
mkdir -p ${ofolder_reg}
mkdir -p ${ofolder_seg}

# Set filenames
file_t1w_mts="${SUBJECT}_acq-T1w_MTS"
file_mton="${SUBJECT}_acq-MTon_MTS"
file_mtoff="${SUBJECT}_acq-MToff_MTS"
file_t2w="${SUBJECT}_T2w"
file_t2s="${SUBJECT}_T2star"
file_t1w="${SUBJECT}_T1w"

# Check if manual segmentation already exists. If it does, copy it locally. If it does not, perform seg.
segment_if_does_not_exist(){
  local file="$1"
  local contrast="$2"
  if [ -e "${PATH_SEGMANUAL}/${SITE}/${file}_seg-manual.nii.gz" ]; then
    rsync -avzh "${PATH_SEGMANUAL}/${SITE}/${file}_seg-manual.nii.gz" ${file}_seg.nii.gz
  else
    # Segment spinal cord
    sct_deepseg_sc -i ${file}.nii.gz -c $contrast -qc ${PATH_QC}
  fi
  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
}

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# Work in a local temp directory (more convenient)
mkdir tmp; cd tmp

# Copy images from source database
rsync -avzh ${PATH_IN}/${file_t1w_mts}.nii.gz .
rsync -avzh ${PATH_IN}/${file_mton}.nii.gz .
rsync -avzh ${PATH_IN}/${file_mtoff}.nii.gz .
rsync -avzh ${PATH_IN}/${file_t2w}.nii.gz .
rsync -avzh ${PATH_IN}/${file_t2s}.nii.gz .
rsync -avzh ${PATH_IN}/${file_t1w}.nii.gz .

# Crop to avoid imperfect slab profile at the edge (altered contrast)
sct_crop_image -i ${file_t1w_mts}.nii.gz -o ${file_t1w_mts}_crop.nii.gz -start 3 -end -3 -dim 2
file_t1w_mts="${file_t1w_mts}_crop"

# Resample to fixed resolution
sct_resample -i ${file_t1w_mts}.nii.gz -o ${file_t1w_mts}_r.nii.gz -mm 0.5x0.5x5 -x spline
file_t1w_mts="${file_t1w_mts}_r"

# Segment spinal cord on T1w scan (only if it does not exist)
segment_if_does_not_exist $file_t1w_mts "t1"
file_seg=$FILESEG

# Create mask
sct_create_mask -i ${file_t1w_mts}.nii.gz -p centerline,${file_seg}.nii.gz -size 55mm -o ${file_t1w_mts}_mask.nii.gz

# Image-based registrations of MToff and MTon to T1w_MTS scan
sct_register_multimodal -i ${file_mtoff}.nii.gz -d ${file_t1w_mts}.nii.gz -dseg ${file_seg}.nii.gz -m ${file_t1w_mts}_mask.nii.gz -param step=1,type=im,algo=slicereg,metric=CC,poly=2 -x spline
file_mtoff="${file_mtoff}_reg"
sct_register_multimodal -i ${file_mton}.nii.gz -d ${file_t1w_mts}.nii.gz -dseg ${file_seg}.nii.gz -m ${file_t1w_mts}_mask.nii.gz -param step=1,type=im,algo=slicereg,metric=CC,poly=2 -x spline
file_mton="${file_mton}_reg"

# For some vendors, T2s scans are 4D. So we need to average them.
sct_maths -i ${file_t2s}.nii.gz -mean t -o ${file_t2s}_mean.nii.gz
file_t2s="${file_t2s}_mean"

# Put other scans in the same voxel space as the T1w_MTS volume (for subsequent cord segmentation)
sct_register_multimodal -i ${file_t2w}.nii.gz -d ${file_t1w_mts}.nii.gz -identity 1 -x spline
file_t2w="${file_t2w}_reg"
sct_register_multimodal -i ${file_t2s}.nii.gz -d ${file_t1w_mts}.nii.gz -identity 1 -x spline
file_t2s="${file_t2s}_reg"
sct_register_multimodal -i ${file_t1w}.nii.gz -d ${file_t1w_mts}.nii.gz -identity 1 -x spline
file_t1w="${file_t1w}_reg"

# Run segmentation on other scans
segment_if_does_not_exist $file_t2w "t2"
file_seg_t2w=$FILESEG
segment_if_does_not_exist $file_t2s "t2s"
file_seg_t2s=$FILESEG
segment_if_does_not_exist $file_t1w "t1"
file_seg_t1w=$FILESEG

# Registration of T2w, T2s and T1w to T1w_MTS scan using the segmentations
sct_register_multimodal -i ${file_seg_t2w}.nii.gz -d ${file_seg}.nii.gz -param step=1,type=im,algo=slicereg -x linear
sct_register_multimodal -i ${file_seg_t2s}.nii.gz -d ${file_seg}.nii.gz -param step=1,type=im,algo=slicereg -x linear
sct_register_multimodal -i ${file_seg_t1w}.nii.gz -d ${file_seg}.nii.gz -param step=1,type=im,algo=slicereg -x linear

# Apply warping field to native files (to avoid 2x interpolation) -- use bspline interpolation
sct_apply_transfo -i ${SUBJECT}_T2w.nii.gz -d ${file_t1w_mts}.nii.gz -w warp_${file_seg_t2w}2${file_seg}.nii.gz
sct_apply_transfo -i ${SUBJECT}_T2star.nii.gz -d ${file_t1w_mts}.nii.gz -w warp_${file_seg_t2s}2${file_seg}.nii.gz
sct_apply_transfo -i ${SUBJECT}_T1w.nii.gz -d ${file_t1w_mts}.nii.gz -w warp_${file_seg_t1w}2${file_seg}.nii.gz -o ${SUBJECT}_T1w_reg.nii.gz

# Average all segmentations together and then binarize. Note: we do not include the T2s because it only has 15 slices
sct_image -i ${file_seg}.nii.gz,${file_seg_t1w}_reg.nii.gz,${file_seg_t2w}_reg.nii.gz -concat t -o tmp.concat.nii.gz
sct_maths -i tmp.concat.nii.gz -mean t -o tmp.concat_mean.nii.gz
sct_maths -i tmp.concat_mean.nii.gz -bin 0.5 -o ${file_t1w_mts}_seg-manual.nii.gz

# TODO: QC

# Verify presence of output files and write log file if error
FILES_TO_CHECK=(
  "${file_seg}.nii.gz"
  "${file_seg_t2w}.nii.gz"
  "${file_seg_t2s}.nii.gz"
  "${file_seg_t1w}.nii.gz"
  "${file_t1w_mts}_seg-manual.nii.gz"
)
for file in ${FILES_TO_CHECK[@]}; do
  if [ ! -e $file ]; then
    echo "${SITE}/${file} does not exist" >> $PATH_LOG/_error_prepare_data.log
  fi
done

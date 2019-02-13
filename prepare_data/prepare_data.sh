#!/bin/bash
#
# Generate segmentation and co-register all multimodal data.
#
# Usage:
#   ./prepare_data.sh <subject_ID> <output path>
#
# Where subject_ID refers to the subject ID according to the BIDS format.
#
# Example:
#   ./prepare_data.sh sub-03 /users/jondoe/bids_data_results/site-01
#

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
sub=$1
PATH_OUTPUT_wSITE=$2

# Create BIDS architecture
PATH_IN="`pwd`/${sub}/anat"
ofolder_seg="${PATH_OUTPUT_wSITE}/derivatives/labels/${sub}/anat"
ofolder_reg="${PATH_OUTPUT_wSITE}/${sub}/anat"
mkdir -p ${ofolder_reg}
mkdir -p ${ofolder_seg}

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# Set filenames
file_t1w_mts="${sub}_acq-T1w_MTS"
file_mton="${sub}_acq-MTon_MTS"
file_mtoff="${sub}_acq-MToff_MTS"
file_t2w="${sub}_T2w"
file_t2s="${sub}_T2star"
file_t1w="${sub}_T1w"

# Copy reference volume
rsync -avzh ${PATH_IN}/${file_t1w_mts}.nii.gz .

# Crop to because image quality is not good at the edge
sct_crop_image -i ${file_t1w_mts}.nii.gz -o ${file_t1w_mts}_crop.nii.gz -start 3 -end -3 -dim 2
file_t1w_mts="${file_t1w_mts}_crop"

# Check if manual segmentation already exists
if [ -e "${ofolder_seg}/${file_t1w_mts}_seg_manual.nii.gz" ]; then
  file_seg="${file_t1w_mts}_seg_manual"
else
  # Segment spinal cord
  sct_deepseg_sc -i ${file_t1w_mts}.nii.gz -c t1 -ofolder ${ofolder_seg}
  file_seg="${file_t1w_mts}_seg"
fi

# Create mask
sct_create_mask -i ${file_t1w_mts}.nii.gz -p centerline,"${ofolder_seg}/${file_seg}.nii.gz" -size 55mm -o ${file_t1w_mts}_mask.nii.gz

# Image-based registrations of MToff and MTon to T1w_MTS scan
sct_register_multimodal -i ${PATH_IN}/${file_mtoff}.nii.gz -d ${file_t1w_mts}.nii.gz -m ${file_t1w_mts}_mask.nii.gz -param step=1,type=im,algo=slicereg,metric=CC,poly=2 -x spline
file_mtoff="${file_mtoff}_reg"
sct_register_multimodal -i ${PATH_IN}/${file_mton}.nii.gz -d ${file_t1w_mts}.nii.gz -m ${file_t1w_mts}_mask.nii.gz -param step=1,type=im,algo=slicereg,metric=CC,poly=2 -x spline
file_mton="${file_mton}_reg"

# Put other scans in the same voxel space as the T1w_MTS volume (for subsequent cord segmentation)
sct_register_multimodal -i ${PATH_IN}/${file_t2w}.nii.gz -d ${file_t1w_mts}.nii.gz -identity 1 -x spline
file_t2w="${file_t2w}_reg"
sct_register_multimodal -i ${PATH_IN}/${file_t2s}.nii.gz -d ${file_t1w_mts}.nii.gz -identity 1 -x spline
file_t2s="${file_t2s}_reg"
sct_register_multimodal -i ${PATH_IN}/${file_t1w}.nii.gz -d ${file_t1w_mts}.nii.gz -identity 1 -x spline
file_t1w="${file_t1w}_reg"

# Run segmentation on other scans
if [ -e "${ofolder_seg}/${file_t2w}_seg_manual.nii.gz" ]; then
  file_seg_t2w="${file_t2w}_seg_manual"
else
  # Segment spinal cord
  sct_deepseg_sc -i ${file_t2w}.nii.gz -c t2 -ofolder ${ofolder_seg}
  file_seg_t2w="${file_t2w}_seg"
fi
# T2s
if [ -e "${ofolder_seg}/${file_t2s}_seg_manual.nii.gz" ]; then
  file_seg_t2s="${file_t2s}_seg_manual"
else
  # Segment spinal cord
  sct_deepseg_sc -i ${file_t2s}.nii.gz -c t2s -ofolder ${ofolder_seg}
  file_seg_t2s="${file_t2s}_seg"
fi
# T1w
if [ -e "${ofolder_seg}/${file_t1w}_seg_manual.nii.gz" ]; then
  file_seg_t1w="${file_t1w}_seg_manual"
else
  # Segment spinal cord
  sct_deepseg_sc -i ${file_t1w}.nii.gz -c t1 -ofolder ${ofolder_seg}
  file_seg_t1w="${file_t1w}_seg"
fi

# Segmentation-based registrations of T2w, T2s and T1w to T1w_MTS scan
sct_register_multimodal -i ${ofolder_seg}/${file_seg_t2w}.nii.gz -d ${ofolder_seg}/${file_seg}.nii.gz -param step=1,type=im,algo=centermass -x linear -ofolder ${ofolder_seg}
sct_register_multimodal -i ${ofolder_seg}/${file_seg_t2s}.nii.gz -d ${ofolder_seg}/${file_seg}.nii.gz -param step=1,type=im,algo=centermass -x linear -ofolder ${ofolder_seg}
sct_register_multimodal -i ${ofolder_seg}/${file_seg_t1w}.nii.gz -d ${ofolder_seg}/${file_seg}.nii.gz -param step=1,type=im,algo=centermass -x linear -ofolder ${ofolder_seg}

# Apply warping field to native files (to avoid 2x interpolation) -- use bspline interpolation
sct_apply_transfo -i ${PATH_IN}/${sub}_T2w.nii.gz -d ${file_t1w_mts}.nii.gz -w ${ofolder_seg}/warp_${file_seg_t2w}2${file_seg}.nii.gz
sct_apply_transfo -i ${PATH_IN}/${sub}_T2star.nii.gz -d ${file_t1w_mts}.nii.gz -w ${ofolder_seg}/warp_${file_seg_t2s}2${file_seg}.nii.gz
sct_apply_transfo -i ${PATH_IN}/${sub}_T1w.nii.gz -d ${file_t1w_mts}.nii.gz -w ${ofolder_seg}/warp_${file_seg_t1w}2${file_seg}.nii.gz

# copying the json files and renaming them :
rsync -avzh ${PATH_IN}/${sub}_acq-T1w_MTS.json ${file_t1w_mts}.json
rsync -avzh ${PATH_IN}/${sub}_acq-MTon_MTS.json ${file_mton}.json
rsync -avzh ${PATH_IN}/${sub}_acq-MToff_MTS.json ${file_mtoff}.json
rsync -avzh ${PATH_IN}/${sub}_T2w.json ${file_t2w}.json
rsync -avzh ${PATH_IN}/${sub}_T2star.json ${file_t2s}.json
rsync -avzh ${PATH_IN}/${sub}_T1w.json ${file_t1w}.json
rsync -avzh ${PATH_IN}/../../dataset_description.json ../../
rsync -avzh ${PATH_IN}/../../participants.json ../../
rsync -avzh ${PATH_IN}/../../participants.tsv ../../

# Delete temporary files (they interfer with the BIDS wrapper)
rm *_mask.nii.gz
rm warp*
rm *crop_reg*
rm ${sub}_acq-T1w_MTS.nii.gz
rm ${ofolder_seg}/warp*
rm ${ofolder_seg}/${sub}_T1w_reg_seg.nii.gz
rm ${ofolder_seg}/${sub}_acq-T1w_MTS_crop_seg_reg.nii.gz
rm ${ofolder_seg}/${sub}_T2star_reg_seg.nii.gz
rm ${ofolder_seg}/${sub}_T2w_reg_seg.nii.gz

# Average all segmentations together. Note: we do not include the T2s because it only has 15 slices
sct_image -i ${ofolder_seg}/${sub}_acq-T1w_MTS_crop_seg.nii.gz,${ofolder_seg}/${sub}_T1w_reg_seg_reg.nii.gz,${ofolder_seg}/${sub}_T2w_reg_seg_reg.nii.gz -concat t -o ${ofolder_seg}/tmp.concat.nii.gz
sct_maths -i ${ofolder_seg}/tmp.concat.nii.gz -mean t -o ${ofolder_seg}/tmp.concat_mean.nii.gz
sct_maths -i ${ofolder_seg}/tmp.concat_mean.nii.gz -bin 0.5 -o ${ofolder_seg}/${file_t1w_mts}_seg-manual.nii.gz
# Cleaning
rm ${ofolder_seg}/${sub}_*reg.*
rm ${ofolder_seg}/${sub}_*seg.*
rm ${ofolder_seg}/tmp.*
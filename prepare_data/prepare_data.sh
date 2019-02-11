#!/bin/bash
#
# Generate segmentation and co-register all multimodal data.
#
# Usage:
#   ./prepare_data.sh <subject_ID>
#
# Where subject_ID refers to the subject ID according to the BIDS format.
#
# Example:
#   ./prepare_data.sh sub-03
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
sct_register_multimodal -i ${PATH_IN}/${file_mton}.nii.gz -d ${file_t1w_mts}.nii.gz -m ${file_t1w_mts}_mask.nii.gz -param step=1,type=im,algo=slicereg,metric=CC,poly=2 -x spline

# Put other scans in the same voxel space as the T1w_MTS volume (for subsequent cord segmentation)
sct_register_multimodal -i ${PATH_IN}/${file_t2w}.nii.gz -d ${file_t1w_mts}.nii.gz -identity 1 -x spline
file_t2w="${file_t2w}_reg"
sct_register_multimodal -i ${PATH_IN}/${file_t2s}.nii.gz -d ${file_t1w_mts}.nii.gz -identity 1 -x spline
sct_register_multimodal -i ${PATH_IN}/${file_t1w}.nii.gz -d ${file_t1w_mts}.nii.gz -identity 1 -x spline

# Run segmentation on other scans
if [ -e "${ofolder_seg}/${file_t2w}_seg_manual.nii.gz" ]; then
  file_seg_t2w="${file_t2w}_seg_manual"
else
  # Segment spinal cord
  sct_deepseg_sc -i ${file_t2w}_reg.nii.gz -c t2 -ofolder ${ofolder_seg}
  file_seg_t2w="${file_t2w}_seg"
fi

# Segmentation-based registrations of T2w, T2s and T1w to T1w_MTS scan
sct_register_multimodal -i ${ofolder_seg}/${file_seg_t2w}.nii.gz -d ${ofolder_seg}/${file_seg}.nii.gz -param step=1,type=im,algo=slicereg,metric=MeanSquares,poly=3 -x linear

# 

# TODO: do the same for T1w and T2s

# TODO: average all segmentations together.


# # copying the T1w_mts file, which everything else is registered into :
# rsync -avzh "${file_t1w_mts}.nii.gz" ${ofolder_reg}/

# # Cropping images to remove first 3 and last 3 slices :

# sct_crop_image -i ${ofolder_reg}/${file_mton}_reg.nii.gz -o ${ofolder_reg}/${file_mton}_reg_crop.nii.gz -start 3 -end -3 -dim 2
# sct_crop_image -i ${ofolder_reg}/${file_mtoff}_reg.nii.gz -o ${ofolder_reg}/${file_mtoff}_reg_crop.nii.gz -start 3 -end -3 -dim 2
# sct_crop_image -i ${ofolder_reg}/${file_t2w}_reg.nii.gz -o ${ofolder_reg}/${file_t2w}_reg_crop.nii.gz -start 3 -end -3 -dim 2
# sct_crop_image -i ${ofolder_reg}/${file_t2s}_reg.nii.gz -o ${ofolder_reg}/${file_t2s}_reg_crop.nii.gz -start 3 -end -3 -dim 2
# sct_crop_image -i ${ofolder_reg}/${file_t1w}_reg.nii.gz -o ${ofolder_reg}/${file_t1w}_reg_crop.nii.gz -start 3 -end -3 -dim 2
# sct_crop_image -i ${ofolder_reg}/${file_t1w_mts}.nii.gz -o ${ofolder_reg}/${file_t1w_mts}_crop.nii.gz -start 3 -end -3 -dim 2 #not registered

# # Delete useless images
# # rm "${ofolder_reg}/${file_t1w_mts}_mask.nii.gz"
# # rm "${ofolder_reg}/${file_t1w_mts}_reg.nii.gz"
# # rm *image_in_RPI_resampled*
# # rm ${ofolder_reg}/*warp* #delete warping fields
# # rm ${ofolder_reg}/*_reg.nii.gz #delete "registered but not cropped" images
# # rm ${ofolder_reg}/${file_t1w_mts}.nii.gz


# # copying the json files and renaming them :
# rsync -avzh "${file_t1w_mts}.json" ${ofolder_reg}/
# mv ${ofolder_reg}/${file_t1w_mts}.json ${ofolder_reg}/${file_t1w_mts}_crop.json #not registered
# rsync -avzh "${file_mton}.json" ${ofolder_reg}/
# mv ${ofolder_reg}/${file_mton}.json ${ofolder_reg}/${file_mton}_reg_crop.json
# rsync -avzh "${file_mtoff}.json" ${ofolder_reg}/
# mv ${ofolder_reg}/${file_mtoff}.json ${ofolder_reg}/${file_mtoff}_reg_crop.json
# rsync -avzh "${file_t2w}.json" ${ofolder_reg}/
# mv ${ofolder_reg}/${file_t2w}.json ${ofolder_reg}/${file_t2w}_reg_crop.json
# rsync -avzh "${file_t2s}.json" ${ofolder_reg}/
# mv ${ofolder_reg}/${file_t2s}.json ${ofolder_reg}/${file_t2s}_reg_crop.json
# rsync -avzh "${file_t1w}.json" ${ofolder_reg}/
# mv ${ofolder_reg}/${file_t1w}.json ${ofolder_reg}/${file_t1w}_reg_crop.json

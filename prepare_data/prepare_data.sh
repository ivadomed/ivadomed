#!/bin/bash
#
# Generate segmentation and co-register all multimodal data.
#
# Usage:
#   ./generate_ground_truth.sh <subject_ID>
#
# Where subject_ID refers to the subject ID according to the BIDS format.
#
# Example:
#   ./generate_ground_truth.sh sub-03
#

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
sub=$1
PATH_OUTPUT_wSITE=$2

# Create BIDS architecture
mkdir -p ${PATH_OUTPUT_wSITE}/${sub}/anat
mkdir -p ${PATH_OUTPUT_wSITE}/derivatives/labels/${sub}/anat
ofolder_seg="${PATH_OUTPUT_wSITE}/derivatives/labels/${sub}/anat"
ofolder_reg="${PATH_OUTPUT_wSITE}/${sub}/anat"

# Go to anat folder where all structural data are located
cd ${sub}/anat


# Filenames
# ==============================================================================
file_t1w_mts="${sub}_acq-T1w_MTS"
file_mton="${sub}_acq-MTon_MTS"
file_mtoff="${sub}_acq-MToff_MTS"
file_t2w="${sub}_T2w"
file_t2s="${sub}_T2star"
file_t1w="${sub}_T1w"
# Check if manual segmentation already exists
if [ -e "${file_t1w}_seg_manual.nii.gz" ]; then
  file_seg="${file_t1w}_seg_manual"
else
  # Segment spinal cord
  sct_deepseg_sc -i ${file_t1w_mts}.nii.gz -c t1  -ofolder ${ofolder_seg}
  file_seg="${file_t1w_mts}_seg"
fi

# Create mask
sct_create_mask -i ${file_t1w_mts}.nii.gz -p centerline,"${ofolder_seg}/${file_seg}.nii.gz" -size 55mm -o ${ofolder_reg}/${file_t1w_mts}_mask.nii.gz

# Registrations to T1w MTS :
# Tips: here we only use rigid transformation because both images have very similar sequence parameters. We don't want to use SyN/BSplineSyN to avoid introducing spurious deformations.
# MToff
sct_register_multimodal -i ${file_mtoff}.nii.gz -d ${file_t1w_mts}.nii.gz -m ${ofolder_reg}/${file_t1w_mts}_mask.nii.gz -param step=1,type=im,algo=slicereg,metric=CC,poly=2 -x spline -ofolder ${ofolder_reg}
# MTon
sct_register_multimodal -i ${file_mton}.nii.gz -d ${file_t1w_mts}.nii.gz -m ${ofolder_reg}/${file_t1w_mts}_mask.nii.gz -param step=1,type=im,algo=slicereg,metric=CC,poly=2 -x spline -ofolder ${ofolder_reg}
# T2w
sct_register_multimodal -i ${file_t2w}.nii.gz -d ${file_t1w_mts}.nii.gz -m ${ofolder_reg}/${file_t1w_mts}_mask.nii.gz -param step=1,type=im,algo=slicereg,metric=CC,poly=2 -x spline -ofolder ${ofolder_reg}
# T2star
sct_register_multimodal -i ${file_t2s}.nii.gz -d ${file_t1w_mts}.nii.gz -m ${ofolder_reg}/${file_t1w_mts}_mask.nii.gz -param step=1,type=im,algo=slicereg,metric=CC,poly=2 -x spline -ofolder ${ofolder_reg}
# T1w
sct_register_multimodal -i ${file_t1w}.nii.gz -d ${file_t1w_mts}.nii.gz -m ${ofolder_reg}/${file_t1w_mts}_mask.nii.gz -param step=1,type=im,algo=slicereg,metric=CC,poly=2 -x spline -ofolder ${ofolder_reg}

# copying the T1w_mts file, which everything else is registered into :
rsync -avzh "${file_t1w_mts}.nii.gz" ${ofolder_reg}/

# Cropping images to remove first 3 and last 3 slices :

sct_crop_image -i ${ofolder_reg}/${file_mton}_reg.nii.gz -o ${ofolder_reg}/${file_mton}_reg_crop.nii.gz -start 3 -end -3 -dim 2
sct_crop_image -i ${ofolder_reg}/${file_mtoff}_reg.nii.gz -o ${ofolder_reg}/${file_mtoff}_reg_crop.nii.gz -start 3 -end -3 -dim 2
sct_crop_image -i ${ofolder_reg}/${file_t2w}_reg.nii.gz -o ${ofolder_reg}/${file_t2w}_reg_crop.nii.gz -start 3 -end -3 -dim 2
sct_crop_image -i ${ofolder_reg}/${file_t2s}_reg.nii.gz -o ${ofolder_reg}/${file_t2s}_reg_crop.nii.gz -start 3 -end -3 -dim 2
sct_crop_image -i ${ofolder_reg}/${file_t1w}_reg.nii.gz -o ${ofolder_reg}/${file_t1w}_reg_crop.nii.gz -start 3 -end -3 -dim 2
sct_crop_image -i ${ofolder_reg}/${file_t1w_mts}.nii.gz -o ${ofolder_reg}/${file_t1w_mts}_crop.nii.gz -start 3 -end -3 -dim 2 #not registered

# Delete useless images
# rm "${ofolder_reg}/${file_t1w_mts}_mask.nii.gz"
# rm "${ofolder_reg}/${file_t1w_mts}_reg.nii.gz"
# rm *image_in_RPI_resampled*
# rm ${ofolder_reg}/*warp* #delete warping fields
# rm ${ofolder_reg}/*_reg.nii.gz #delete "registered but not cropped" images
# rm ${ofolder_reg}/${file_t1w_mts}.nii.gz


# copying the json files and renaming them :
rsync -avzh "${file_t1w_mts}.json" ${ofolder_reg}/
mv ${ofolder_reg}/${file_t1w_mts}.json ${ofolder_reg}/${file_t1w_mts}_crop.json #not registered
rsync -avzh "${file_mton}.json" ${ofolder_reg}/
mv ${ofolder_reg}/${file_mton}.json ${ofolder_reg}/${file_mton}_reg_crop.json
rsync -avzh "${file_mtoff}.json" ${ofolder_reg}/
mv ${ofolder_reg}/${file_mtoff}.json ${ofolder_reg}/${file_mtoff}_reg_crop.json
rsync -avzh "${file_t2w}.json" ${ofolder_reg}/
mv ${ofolder_reg}/${file_t2w}.json ${ofolder_reg}/${file_t2w}_reg_crop.json
rsync -avzh "${file_t2s}.json" ${ofolder_reg}/
mv ${ofolder_reg}/${file_t2s}.json ${ofolder_reg}/${file_t2s}_reg_crop.json
rsync -avzh "${file_t1w}.json" ${ofolder_reg}/
mv ${ofolder_reg}/${file_t1w}.json ${ofolder_reg}/${file_t1w}_reg_crop.json

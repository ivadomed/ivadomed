root="/home/GRAMES.POLYMTL.CA/p114001"
data_root="/home/GRAMES.POLYMTL.CA/p114001/Projects/ivadomed/leander/data-multi-subject-hemis"
im_T1="$data_root/sub-juntendo750w02/anat/sub-juntendo750w02_T1w.nii.gz"
im_T2="$data_root/sub-juntendo750w02/anat/sub-juntendo750w02_T2w.nii.gz"

# Apply transformations to annotations
sct_apply_transfo \
-i "$data_root/derivatives/labels/sub-juntendo750w02/anat/sub-juntendo750w02_T1w_RPI_r_seg-manual.nii.gz" \
-d $im_T2 \
-w "$root/registration_experiments/affine/vanilla/warp_sub-juntendo750w02_T1w2sub-juntendo750w02_T2w.nii.gz" \
-o "$root/registration_experiments/affine/registered_anno_nn/sub-juntendo750w02_T1w_RPI_r_seg-manual_reg_vanilla.nii.gz" \
-x nn

sct_apply_transfo \
-i "$data_root/derivatives/labels/sub-juntendo750w02/anat/sub-juntendo750w02_T1w_RPI_r_seg-manual.nii.gz" \
-d $im_T2 \
-w "$root/registration_experiments/affine/vanilla_slicewise/warp_sub-juntendo750w02_T1w2sub-juntendo750w02_T2w.nii.gz" \
-o "$root/registration_experiments/affine/registered_anno_nn/sub-juntendo750w02_T1w_RPI_r_seg-manual_reg_vanilla_slicewise.nii.gz" \
-x nn

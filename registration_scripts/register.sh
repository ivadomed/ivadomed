# Registering HeMIS-multi-subject data on Duke

root="/home/GRAMES.POLYMTL.CA/p114001"
data_root="/home/GRAMES.POLYMTL.CA/p114001/Projects/ivadomed/leander/data-multi-subject-hemis"
im_T1="$data_root/sub-juntendo750w02/anat/sub-juntendo750w02_T1w.nii.gz"
im_T2="$data_root/sub-juntendo750w02/anat/sub-juntendo750w02_T2w.nii.gz"

# Affine, vanilla registration
sct_register_multimodal \
-i $im_T1 \
-d $im_T2 \
-ofolder "$root/registration_experiments/affine/vanilla" \
-qc "$root/registration_experiments/affine/qc_reports" \
-dseg "$root/registration_experiments/sub-juntendo750w02_T2w_seg.nii.gz" \
-param "step=1,type=im,algo=affine,metric=CC,iter=10,smooth=0,gradStep=0.1,slicewise=0"

# # Affine, vanilla registration + slicewise
sct_register_multimodal \
-i $im_T1 \
-d $im_T2 \
-ofolder "$root/registration_experiments/affine/vanilla_slicewise" \
-qc "$root/registration_experiments/affine/qc_reports" \
-dseg "$root/registration_experiments/sub-juntendo750w02_T2w_seg.nii.gz" \
-param "step=1,type=im,algo=affine,metric=CC,iter=10,smooth=0,gradStep=0.1,slicewise=1"

# Affine registration + binary map
sct_register_multimodal \
-i $im_T1 \
-d $im_T2 \
-ofolder "$root/registration_experiments/affine/vanilla_binarymap" \
-qc "$root/registration_experiments/affine/qc_reports" \
-m "$root/registration_experiments/sub-juntendo750w02_T2w_binary_mask.nii.gz" \
-dseg "$root/registration_experiments/sub-juntendo750w02_T2w_seg.nii.gz" \
-param "step=1,type=im,algo=affine,metric=CC,iter=10,smooth=0,gradStep=0.1,slicewise=0"

# # Affine registration + gaussian map
sct_register_multimodal \
-i $im_T1 \
-d $im_T2 \
-ofolder "$root/registration_experiments/affine/vanilla_gaussianmap" \
-qc "$root/registration_experiments/affine/qc_reports" \
-m "$root/registration_experiments/sub-juntendo750w02_T2w_gaussian_mask.nii.gz" \
-dseg "$root/registration_experiments/sub-juntendo750w02_T2w_seg.nii.gz" \
-param "step=1,type=im,algo=affine,metric=CC,iter=10,smooth=0,gradStep=0.1,slicewise=0"

root="/home/GRAMES.POLYMTL.CA/p114001"
data_root="/home/GRAMES.POLYMTL.CA/p114001/Projects/ivadomed/leander/data-multi-subject-hemis"
im_T1="$data_root/sub-juntendo750w02/anat/sub-juntendo750w02_T1w.nii.gz"
im_T2="$data_root/sub-juntendo750w02/anat/sub-juntendo750w02_T2w.nii.gz"

# Getting prerequisite data

# T2 SC segmentation
sct_deepseg_sc \
-i $im_T2 \
-c t2 \
-ofolder "/home/GRAMES.POLYMTL.CA/p114001/registration_experiments/" \
-qc "/home/GRAMES.POLYMTL.CA/p114001/registration_experiments/qc_reports"

# T2 SC centerline
sct_get_centerline \
-i $im_T2 \
-c t2 \
-qc "/home/GRAMES.POLYMTL.CA/p114001/registration_experiments/qc_reports"

# T2 binary mask
sct_create_mask \
-i $im_t2 \
-p centerline,"/home/GRAMES.POLYMTL.CA/p114001/registration_experiments/sub-juntendo750w02_T2w_centerline.nii.gz" \
-o "/home/GRAMES.POLYMTL.CA/p114001/registration_experiments/sub-juntendo750w02_T2w_binary_mask.nii.gz"

# T2 gaussian mask
sct_create_mask \
-i $im_t2 \
-p centerline,"/home/GRAMES.POLYMTL.CA/p114001/registration_experiments/sub-juntendo750w02_T2w_centerline.nii.gz" \
-o "/home/GRAMES.POLYMTL.CA/p114001/registration_experiments/sub-juntendo750w02_T2w_gaussian_mask.nii.gz" \
-f gaussian

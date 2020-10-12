#!/bin/bash

# Randomly selected image
root="/home/leander/Documents/BME_y3_q1/internship/data/"
im_T1="$root/data-multi-subject/sub-cardiff01/anat/sub-cardiff01_T1w.nii.gz"
im_T2="$root/data-multi-subject/sub-cardiff01/anat/sub-cardiff01_T2w.nii.gz"

# Get SC segmentations
sct_deepseg_sc \
-i $im_T1 \
-c t1 \
-ofolder "$root/data-multi-subject-registered/" \
-qc "$root/data-multi-subject-registered/qc_reports/" 

sct_deepseg_sc \
-i $im_T2 \
-c t2 \
-ofolder "$root/data-multi-subject-registered/" \
-qc "$root/data-multi-subject-registered/qc_reports/" 

# Get vertebrae labels
sct_label_vertebrae \
-i $im_T1 \
-s "$root/data-multi-subject-registered/sub-cardiff01_T1w_seg.nii.gz" \
-c t1 \
-qc "$root/data-multi-subject-registered/qc_reports/"

sct_label_vertebrae \
-i $im_T2 \
-s "$root/data-multi-subject-registered/sub-cardiff01_T2w_seg.nii.gz" \
-c t2 \
-qc "$root/data-multi-subject-registered/qc_reports/"

# Register T1 to T2
# New lines are for readability, not in the actual script
sct_register_multimodal \
-i $im_T1 \
-d $im_T2 \
-o "$root/data-multi-subject-registered/sub-cardiff01_T1w_reg.nii.gz" \
-qc "$root/data-multi-subject-registered/qc_reports" \
-iseg "$root/data-multi-subject-registered/sub-cardiff01_T1w_seg.nii.gz" \
-dseg "$root/data-multi-subject-registered/sub-cardiff01_T2w_seg.nii.gz" \
-ilabel "$root/data-multi-subject-registered/sub-cardiff01_T1w_seg_labeled_discs.nii.gz" \
-dlabel "$root/data-multi-subject-registered/sub-cardiff01_T2w_seg_labeled_discs.nii.gz" \
-param "step=0,type=label,dof=Tx_Ty_Tz_Sz:  
        step=1,type=imseg,algo=centermassrot,metric=MeanSquares,iter=10,smooth=0,gradStep=0.5,slicewise=0,smoothWarpXY=2,pca_eigenratio_th=1.6:
        step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,smooth=1,gradStep=0.5,slicewise=0,smoothWarpXY=2,pca_eigenratio_th=1.6"

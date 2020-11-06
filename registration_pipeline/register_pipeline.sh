#!/bin/bash

# --PREAMBLE--

# Hypothethical call of sct_run_batch
# sct_run_batch -config config_file.json

# Hypothetical JSON file
# {
#     "path_data"   : "~/sct_data"
#     "path_output" : "~/pipeline_results"
#     "script"      : "nature_paper_analysis.sh"
#     "jobs"        : -1
# }

# --!PREAMBLE--

# Pipeline script for mass T1-to-T2 registration of spinal cord MRI scans
# with CSF and SC annotations. Quantifies registration quality /w dice overlap.

# Steps:
#     * Read/find SC segmentation T1/T2 - if not available, make them
#     * Register T1 -> T2 with input parameters
#     * Apply output warp field to T1 SC segmentation
#     * Compute dice(T2_SC_seg, T1_SC_seg_reg)
#     * Output to CSV (small python script?)

# Input params
SUBJECT=$1
PATH_DATA="/home/GRAMES.POLYMTL.CA/p114001/data_nvme_p114001/data-multi-subject-hemis" # TODO: REMOVE (only for testing)
PATH_QC="/home/GRAMES.POLYMTL.CA/p114001/data_nmve_p114010/registration_pipeline/qc_reports" # TODO: REMOVE (only for testing)

# FUNCTIONS
# ===========

# Check if SC segmentation of T1/T2 anatomical scans exist.
# If exists, copy local - if not, perform segmentation.
segment_if_does_not_exist(){

    # Input
    local file="$1"
    local contrast="$2"

    # Manage file names/variables
    local seg_suffix="RPI_r_seg-manual"
    FILE_SEG="${file}_seg"
    FILE_SEG_MANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_${seg_suffix}.nii.gz"
    local FILE="${PATH_DATA}/${SUBJECT}/anat/${file}.nii.gz"

    # Look for SC segmentation, make if not found
    echo
    echo "Looking for SC segmentation: $FILE_SEG_MANUAL"
    if [[ -e $FILE_SEG_MANUAL ]]; then
        echo "Found! Using existing SC segmentation."
        rsync -avzh $FILE_SEG_MANUAL ${FILE_SEG}.nii.gz
        sct_qc -i $FILE -s ${FILE_SEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
        echo "Not found! Performing automatic SC segmentation."
        sct_deepseg_sc -i $FILE -c $contrast -qc ${PATH_QC} -qc-subject ${SUBJECT}
    fi
}

# SCRIPT
# =========

# e.g. SUBJECT="sub-juntendo750w02"
file_t1="${SUBJECT}_T1w"
file_t2="${SUBJECT}_T2w"

# Get T1/T2 spinal cord segmentations pre-registration
segment_if_does_not_exist $file_t1 "t1"
segment_if_does_not_exist $file_t2 "t2"

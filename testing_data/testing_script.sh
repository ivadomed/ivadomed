#!/bin/bash
echo "convert_onnx"
ivadomed_convert_to_onnx -m testing_data/model_unet_test.pt -d 2
echo "extract_small"
ivadomed_extract_small_dataset -i testing_data/ -o ./small_dataset -n 10 -c T1w,T2w -d 0
echo "visualization"
ivadomed_visualize_transforms -i testing_data/sub-test001/anat/sub-test001_T1w.nii.gz -n 1 -c testing_data/model_config.json -r testing_data/derivatives/labels/sub-test001/anat/sub-test001_T1w_seg-manual.nii.gz
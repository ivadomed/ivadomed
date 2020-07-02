#!/bin/bash
echo "convert_onnx"
ivadomed_convert_to_onnx -m model_unet_test.pt -d 2
echo "extract_small"
ivadomed_extract_small_dataset -i ./ -o ./small_dataset -n 10 -c T1w,T2w -d 0
# Data preparation

These scripts prepare the data for training. It takes as input the [Spinal Cord MRI Public Database](https://osf.io/76jkx/) and outputs BIDS-compatible datasets with segmentation labels for each subject. More specifically, for each subject, the segmentation is run in one volume (T1w), then all volumes are registered to the T1w volume so that all volumes are in the same voxel space and the unique segmentation can be used across volumes.

## Dependencies

In its current state, this pipeline uses [SCT development version](https://github.com/neuropoly/spinalcordtoolbox#install-from-github-development). Once the pipeline is finalized, a stable version of SCT will be associated with this pipeline and indicated here. For now, please use the latest development version of SCT.

## How to run

- Copy the file `parameters_template.sh` and rename it as `parameters.sh`.
- Edit the file `parameters.sh` and modify the variables according to your needs.
- Make sure input files are present: `./run_process.sh check_input_files.sh`
- Process data: `./run_process.sh prepare_data.sh`

**Perform QC:**
- Open qc/index.html
- Search only for "deepseg" QC entries (use "search" field)
- Take a screenshot of the browser when you spot a problem (wait for the segmentation to appear before taking the screenshot)
- If the data are of **very** bad quality, also take a screenshot (this time, wait for the segmentation to disappear)
- Copy all screenshots under qc_feedback/

**Manually correct the segmentations:**

Check the following files:

| Image  | Segmentation  |
|:---|:---|
| sub-XX_acq-T1w_MTS_crop_r.nii.gz | sub-XX_acq-T1w_MTS_crop_r_seg.nii.gz|
| sub-XX_T1w_reg.nii.gz | sub-XX_T1w_reg_seg.nii.gz |
| sub-XX_T2w_reg.nii.gz | sub-XX_T2w_reg_seg.nii.gz |
| sub-XX_T2star_mean_reg.nii.gz | sub-XX_T2star_mean_reg_seg.nii.gz |

- Open the segmentation with `fsleyes`
- Manually correct it
- Save with suffix `-manual`.
- Move to a folder named seg_manual/$SITE/$FILENAME. E.g.: `spineGeneric_201903031331/seg_manual/amu_spineGeneric/sub-01_acq-T1w_MTS_crop_r_seg-manual.nii.gz`

Once QC and manual correction is done, re-run processing and then delete useless files, copy json sidecars, move segmentations to derivatives/: `./run_process.sh delete_temp_files.sh`

#!/usr/bin/env python
"""
This script applies a trained model on a single image.

Usage:

```
python scripts/segment_image.py -m <FOLDER_MODEL> -i <IMAGE>
```
Where:
- FOLDER_MODEL: Path to folder that contains ONNX and/or PT model and ivadomed JSON config file.
- IMAGE: Path to image in NIfTI or PNG format.
"""

# TODO: create entry_points in setup.py and update docstrings usage
# TODO: 'add_suffix' and 'splitext' should be moved to utils library (if it makes sense).

import os
import nibabel as nib

from ivadomed import inference as imed_inference


# TODO: make it input params
path_model = "/Users/julien/temp/rosenberg_nvme/model_seg_lesion_mp2rage_20230102_145722/model_seg_lesion_mp2rage"
input_filenames = ["/Users/julien/data.neuro/basel-mp2rage/sub-P005/anat/sub-P005_UNIT1.nii.gz"]


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension.

    :param fname: absolute or relative file name. Example: t2.nii
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii

    Examples:
    .. code:: python

        add_suffix(t2.nii, _mean) -> t2_mean.nii
        add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return stem + suffix + ext


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    dir_, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            break
    else:
        stem, ext = os.path.splitext(filename)

    return os.path.join(dir_, stem), ext

# options = {"pixel_size": [0.13, 0.13], "overlap_2D": [48, 48], "binarize_maxpooling": True}
# TODO: the 'no_patch' option does not seem to work as expected, because only a fraction of the image is segmented.
options = {"no_patch": True}

nii_lst, target_lst = imed_inference.segment_volume(path_model, input_filenames, options=options)

for i in range(len(nii_lst)):
    nib.save(nii_lst[i], add_suffix(input_filenames[i], '_pred'))

# TODO: add to support PNG
# imed_inference.pred_to_png(nii_lst, target_lst, "<path_to_the_source_file>/image")

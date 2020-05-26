import os
import json
import nibabel as nib
import numpy as np

from ivadomed import  utils as imed_utils
from ivadomed import postprocessing as imed_postpro
from ivadomed.loader import utils as imed_loader_utils


def get_bounding_boxes(mask):
    """
    Generates a 3D bounding box around a given mask
    :param mask: numpy array with the mask of the ROI
    :return: bounding box coordinate (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    coords = np.where(mask != 0)
    dimensions = []
    for i in range(len(coords)):
        dimensions.append(int(coords[i].min()))
        dimensions.append(int(coords[i].max()))

    return dimensions


def adjust_bb_size(bounding_box, factor, multiple_16, resample=False):
    coord = []
    for i in range(len(bounding_box) // 2):
        d_min, d_max = bounding_box[2 * i: (2 * i) + 2]
        if resample:
            d_min, d_max_ = d_min * factor[i], d_max * factor[i]
            dim_len = d_max - d_min
        else:
            dim_len = (d_max - d_min) * factor[i]

        if multiple_16:
            padding = (16 - dim_len % 16) if (16 - dim_len % 16) != 16 else 0
            dim_len = dim_len + padding
        # new min and max coordinates
        min_coord = d_min - (dim_len - (d_max - d_min)) // 2
        coord.append(int(round(max(min_coord, 0))))
        coord.append(int(coord[-1] + dim_len))
        if multiple_16:
            assert (coord[-1] - coord[-2]) % 16 == 0

    return coord


def generate_bounding_box_file(subject_list, model_path, log_dir, gpu_number=0, slice_axis=0, keep_largest_only=True,
                               multiple_16=True, safety_factor=None):
    bounding_box_dict = {}
    if safety_factor is None:
        safety_factor = [1.0, 1.0, 1.0]
    for subject in subject_list:
        subject_path = str(subject.record["absolute_path"])
        object_mask = imed_utils.segment_volume(model_path, subject_path, gpu_number=gpu_number)
        if keep_largest_only:
            object_mask = imed_postpro.keep_largest_object(object_mask)
        ras_orientation = nib.as_closest_canonical(object_mask)
        hwd_orientation = imed_loader_utils.orient_img_hwd(ras_orientation.get_fdata()[..., 0], slice_axis)
        bounding_box = get_bounding_boxes(hwd_orientation)
        bounding_box_dict[subject_path] = adjust_bb_size(bounding_box, safety_factor, multiple_16)

    file_path = os.path.join(log_dir, 'bounding_boxes.json')
    with open(file_path, 'w') as fp:
        json.dump(bounding_box_dict, fp, indent=4)
    return bounding_box_dict


def resample_bounding_box(metadata, resample, multiple_16=True):
    hspace, wspace, dspace = resample
    hfactor = metadata['input_metadata'][0]['zooms'][0] / hspace
    wfactor = metadata['input_metadata'][0]['zooms'][1] / wspace
    dfactor = metadata['input_metadata'][0]['zooms'][2] / dspace
    factor = (hfactor, wfactor, dfactor)
    coord = adjust_bb_size(metadata['input_metadata'][0]['bounding_box'], factor, multiple_16, resample=True)

    for i in range(len(metadata['input_metadata'])):
        metadata['input_metadata'][i]['bounding_box'] = coord

    for i in range(len(metadata['input_metadata'])):
        metadata['gt_metadata'][i]['bounding_box'] = coord

import json
import os
import statistics

import nibabel as nib
import numpy as np
from scipy import ndimage

from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils


def get_bounding_boxes(mask):
    """
    Generates a 3D bounding box around a given mask.
    :param mask: numpy array with the mask of the ROI
    :return: bounding box coordinate (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    # Label the different objects in the mask
    labeled_mask, _ = ndimage.measurements.label(mask)
    object_labels = np.unique(labeled_mask)
    bounding_boxes = []
    for label in object_labels[1:]:
        single_object = labeled_mask == label
        coords = np.where(single_object)
        dimensions = []
        for i in range(len(coords)):
            dimensions.append(int(coords[i].min()))
            dimensions.append(int(coords[i].max()))
        bounding_boxes.append(dimensions)

    return bounding_boxes


def adjust_bb_size(bounding_box, factor, resample=False):
    coord = []
    for i in range(len(bounding_box) // 2):
        d_min, d_max = bounding_box[2 * i: (2 * i) + 2]
        if resample:
            d_min, d_max = d_min * factor[i], d_max * factor[i]
            dim_len = d_max - d_min
        else:
            dim_len = (d_max - d_min) * factor[i]

        # new min and max coordinates
        min_coord = d_min - (dim_len - (d_max - d_min)) // 2
        coord.append(int(round(max(min_coord, 0))))
        coord.append(int(coord[-1] + dim_len))

    return coord


def resize_to_multiple(shape, multiple):
    new_dim = []
    for dim_len, m in zip(shape, multiple):
        padding = (m - dim_len % m) if (m - dim_len % m) != m else 0
        new_dim.append(dim_len + padding)
    return new_dim


def generate_bounding_box_file(subject_list, model_path, log_dir, gpu_number=0, slice_axis=0, contrast_lst=None,
                               keep_largest_only=True, safety_factor=None):
    bounding_box_dict = {}
    if safety_factor is None:
        safety_factor = [1.0, 1.0, 1.0]
    for subject in subject_list:
        if subject.record["modality"] in contrast_lst:
            subject_path = str(subject.record["absolute_path"])
            object_mask = imed_utils.segment_volume(model_path, subject_path, gpu_number=gpu_number)
            if keep_largest_only:
                object_mask = imed_postpro.keep_largest_object(object_mask)

            ras_orientation = nib.as_closest_canonical(object_mask)
            hwd_orientation = imed_loader_utils.orient_img_hwd(ras_orientation.get_fdata()[..., 0], slice_axis)
            bounding_boxes = get_bounding_boxes(hwd_orientation)
            bounding_box_dict[subject_path] = [adjust_bb_size(bb, safety_factor) for bb in bounding_boxes]

    file_path = os.path.join(log_dir, 'bounding_boxes.json')
    with open(file_path, 'w') as fp:
        json.dump(bounding_box_dict, fp, indent=4)
    return bounding_box_dict


def resample_bounding_box(metadata, transform):
    for idx, transfo in enumerate(transform.transform["im"].transforms):
        if "Resample" in str(type(transfo)):
            hspace, wspace, dspace = (transfo.hspace, transfo.wspace, transfo.dspace)
            hfactor = metadata['input_metadata'][0]['zooms'][0] / hspace
            wfactor = metadata['input_metadata'][0]['zooms'][1] / wspace
            dfactor = metadata['input_metadata'][0]['zooms'][2] / dspace
            factor = (hfactor, wfactor, dfactor)
            coord = adjust_bb_size(metadata['input_metadata'][0]['bounding_box'], factor, resample=True)

            for i in range(len(metadata['input_metadata'])):
                metadata['input_metadata'][i]['bounding_box'] = coord

            for i in range(len(metadata['input_metadata'])):
                metadata['gt_metadata'][i]['bounding_box'] = coord
            break


def adjust_transforms(transforms, seg_pair, length=None, stride=None):
    resample_idx = -1
    for img_type in transforms.transform:
        for idx, transfo in enumerate(transforms.transform[img_type].transforms):
            if "BoundingBoxCrop" in str(type(transfo)):
                transforms.transform[img_type].transforms.pop(idx)
            if "Resample" in str(type(transfo)):
                resample_idx = idx

    resample_bounding_box(seg_pair, transforms)
    for img_type in transforms.transform:
        h_min, h_max, w_min, w_max, d_min, d_max = seg_pair['input_metadata'][0]['bounding_box']
        size = [h_max - h_min, w_max - w_min, d_max - d_min]

        if length is not None and stride is not None:
            for idx, dim in enumerate(size):
                if dim < length[idx]:
                    size[idx] = length[idx]
            size = resize_to_multiple(size, stride)
        transform_obj = imed_transforms.BoundingBoxCrop(size=size)
        transforms.transform[img_type].transforms.insert(resample_idx + 1, transform_obj)


def adjust_undo_transforms(transforms, seg_pair):
    for img_type in transforms.transform:
        for idx, transfo in enumerate(transforms.transform[img_type].transforms):
            if "BoundingBoxCrop" in str(type(transfo)):
                transforms.transform[img_type].transforms.pop(idx)
                size = list(seg_pair['input_metadata'][0][0]['index_shape'])
                transform_obj = imed_transforms.BoundingBoxCrop(size=size)
                transforms.transform[img_type].transforms.insert(idx, transform_obj)


def load_bounding_boxes(object_detection_params, subjects, slice_axis, constrast_lst):
    # Load or generate bounding boxes and save them in json file
    bounding_box_dict = {}
    if object_detection_params is None or object_detection_params['object_detection_path'] is None:
        return bounding_box_dict
    bounding_box_path = os.path.join(object_detection_params['log_directory'], 'bounding_boxes.json')
    if os.path.exists(bounding_box_path):
        with open(bounding_box_path, 'r') as fp:
            bounding_box_dict = json.load(fp)
    elif object_detection_params['object_detection_path'] is not None and \
            os.path.exists(object_detection_params['object_detection_path']):
        print("Generating bounding boxes...")
        bounding_box_dict = generate_bounding_box_file(subjects,
                                                       object_detection_params['object_detection_path'],
                                                       object_detection_params['log_directory'],
                                                       object_detection_params['gpu'],
                                                       slice_axis,
                                                       constrast_lst,
                                                       safety_factor=object_detection_params['safety_factor'])
    elif object_detection_params['object_detection_path'] is not None:
        raise RuntimeError("Path to object detection model doesn't exist")

    return bounding_box_dict


def verify_metadata(metadata, has_bounding_box):
    index_has_bounding_box = all(['bounding_box' in metadata['input_metadata'][i]
                                  for i in range(len(metadata['input_metadata']))])
    for gt_metadata in metadata['gt_metadata']:
        if gt_metadata is not None:
            index_has_bounding_box &= 'bounding_box' in gt_metadata

    has_bounding_box &= index_has_bounding_box
    return has_bounding_box


def compute_bb_statistics(bounding_box_path):
    with open(bounding_box_path, 'r') as fp:
        bounding_box_dict = json.load(fp)

    h, w, d, v = [], [], [], []
    for box in bounding_box_dict:
        h_min, h_max, w_min, w_max, d_min, d_max = bounding_box_dict[box]
        h.append(h_max - h_min)
        w.append(w_max - w_min)
        d.append(d_max - d_min)
        v.append((h_max - h_min) * (w_max - w_min) * 2 * (d_max - d_min))

    print('Mean height: {} +/- {}, min: {}, max: {}'.format(statistics.mean(h), statistics.stdev(h), min(h), max(h)))
    print('Mean width: {} +/- {}, min: {}, max: {}'.format(statistics.mean(w), statistics.stdev(w), min(w), max(w)))
    print('Mean depth: {} +/- {}, min: {}, max: {}'.format(statistics.mean(d), statistics.stdev(d), min(d), max(d)))
    print('Mean volume: {} +/- {}, min: {}, max: {}'.format(statistics.mean(v), statistics.stdev(v), min(v), max(v)))

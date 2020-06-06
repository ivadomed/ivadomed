import copy


def get_preprocessing_transforms(transforms):
    original_transforms = copy.deepcopy(transforms)
    preprocessing_transforms = copy.deepcopy(transforms)
    for idx, tr in enumerate(original_transforms):
        if "preprocessing" in transforms[tr] and transforms[tr]["preprocessing"]:
            del transforms[tr]
        else:
            del preprocessing_transforms[tr]

    return preprocessing_transforms


def apply_transforms(transforms, seg_pair, roi_pair=None):
    if transforms is None:
        return (seg_pair, roi_pair)

    if roi_pair is not None:
        stack_roi, metadata_roi = transforms(sample=roi_pair["gt"],
                                             metadata=roi_pair['gt_metadata'],
                                             data_type="roi")
    # Run transforms on images
    stack_input, metadata_input = transforms(sample=seg_pair["input"],
                                             metadata=seg_pair['input_metadata'],
                                             data_type="im")
    # Run transforms on images
    stack_gt, metadata_gt = transforms(sample=seg_pair["gt"],
                                       metadata=seg_pair['gt_metadata'],
                                       data_type="gt")
    seg_pair = {
        'input': stack_input,
        'gt': stack_gt,
        'input_metadata': metadata_input,
        'gt_metadata': metadata_gt
    }

    if roi_pair is not None and len(roi_pair['gt']):
        roi_pair = {
            'input': stack_input,
            'gt': stack_roi,
            'input_metadata': metadata_input,
            'gt_metadata': metadata_roi
        }
    return (seg_pair, roi_pair)

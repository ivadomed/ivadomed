import copy


def get_preprocessing_transforms(transforms):
    original_transforms = copy.deepcopy(transforms)
    preprocessing_transforms = copy.deepcopy(transforms)
    for idx, tr in enumerate(original_transforms):
        if "preprocessing" in transforms[tr] and transforms[tr]["preprocessing"]:
            del transforms[tr]
            del transforms[tr]["preprocessing"]
        else:
            del preprocessing_transforms[tr]

    return preprocessing_transforms
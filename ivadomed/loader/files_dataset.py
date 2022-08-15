from __future__ import annotations
from pathlib import Path
import typing
from typing import List, Tuple

if typing.TYPE_CHECKING:
    from ivadomed.loader.generalized_loader_configuration import (
        GeneralizedLoaderConfiguration,
    )

from ivadomed.loader.mri2d_segmentation_dataset import MRI2DSegmentationDataset

from ivadomed.keywords import (
    ModelParamsKW,
    MetadataKW,
)

example_json: dict = {
    "type": "FILES",
    "subset_label": "Subset3",
    "image_ground_truth": [
        [
            [
                r"sub-02/ses-04/anat/sub-02_ses-04_acq-MToff_MTS.nii",
                r"sub-02/ses-04/anat/sub-02_ses-04_acq-MTon_MTS.nii",
            ],
            [
                "derivatives/labels/sub-02/ses-04/anat/sub-02_ses-04_mt-off_MTS_lesion-manual-rater1.nii"
            ],
        ],
        [
            [
                r"sub-02/ses-05/anat/sub-02_ses-05_acq-MToff_MTS.nii",
                r"sub-02/ses-05/anat/sub-02_ses-05_acq-MTon_MTS.nii",
            ],
            [
                "derivatives/labels/sub-02/ses-05/anat/sub-02_ses-05_mt-off_MTS_lesion-manual-rater1.nii"
            ],
        ],
        [
            [
                r"sub-02/ses-04/anat/sub-02_ses-04_flip-1_mt-off_MTS.nii",
                r"sub-02/ses-04/anat/sub-02_ses-04_flip-1_mt-on_MTS.nii",
            ],
            [
                "derivatives/labels/sub-02/ses-04/anat/sub-02_ses-04_mt-off_MTS_lesion-manual-rater1.nii"
            ],
        ],
        [
            [
                r"sub-02/ses-04/anat/sub-02_ses-04_flip-1_mt-off_MTS.nii",
                r"sub-02/ses-04/anat/sub-02_ses-04_flip-1_mt-on_MTS.nii",
                r"sub-02/ses-04/anat/sub-02_ses-04_flip-2_mt-off_MTS.nii",
                r"sub-02/ses-04/anat/sub-02_ses-04_flip-2_mt-on_MTS.nii",
            ],
            [
                "derivatives/labels/sub-02/ses-04/anat/sub-02_ses-04_mt-off_MTS_lesion-manual-rater1.nii"
            ],
        ],
    ],
    "expected_input": 2,
    "expected_gt": 1,
    "meta_data_csv": "/metadata.csv",  # assumed to be the same shape as the default run.
    "missing_files_handle": "drop_subject",
    "excessive_files_handle": "use_first_and_warn",
    "path_data": r"C:/Temp/Test",
}


class FilesDataset(MRI2DSegmentationDataset):
    """Files Sets specific dataset loader"""

    def __init__(
        self, dict_files_pairing: dict, config: GeneralizedLoaderConfiguration
    ):
        """
        Construator that leverage a generalized loader configuration
        Args:
            dict_files_pairing:
            config:
        """

        # This is the key attribute that needs to be populated once data loading is complete.
        self.filename_pairs: List[Tuple[list, list, str, dict]] = []

        # Create the placeholder metadata dict
        if config.metadata_choice == MetadataKW.MRI_PARAMS:
            self.metadata: dict = {
                "FlipAngle": [],
                "RepetitionTime": [],
                "EchoTime": [],
                "Manufacturer": [],
            }

        # We assume user explicitly provide the subject lists so WE do not do any additional filtering

        # Currently does not support contrast balance (See BIDS Data 2D for reference implementation)
        # Create a dictionary with the number of subjects for each contrast of contrast_balance

        # NOT going to support bounding boxes

        # NOT going to care about multi-contrast/channels as we assume user explicitly provide that.

        # Get all derivatives filenames frm User JSON

        #################
        # Create filename_pairs
        #################
        self.parse_spec_json_and_update_filename_pairs(dict_files_pairing)

        if not self.filename_pairs:
            raise Exception(
                "No subjects were selected - check selection of parameters on config.json"
            )

        length = (
            config.model_params[ModelParamsKW.LENGTH_2D]
            if ModelParamsKW.LENGTH_2D in config.model_params
            else []
        )
        stride = (
            config.model_params[ModelParamsKW.STRIDE_2D]
            if ModelParamsKW.STRIDE_2D in config.model_params
            else []
        )

        # Call the parent class constructor for MRI2DSegmentationDataset
        super().__init__(
            self.filename_pairs,
            length,
            stride,
            config.slice_axis,
            config.nibabel_cache,
            config.transform,
            config.slice_filter_fn,
            config.patch_filter_fn,
            config.task,
            config.roi_params,
            config.soft_gt,
            config.is_input_dropout,
        )

    def parse_spec_json_and_update_filename_pairs(self, loader_json: dict):
        """Load the json file and return the dictionary"""
        # Given a JSON file, try to load the file pairing from it.
        assert loader_json.get("type") == "FILES"

        path_data = loader_json.get("path_data", ".")

        n_expected_input: int = loader_json.get("expected_input", 0)
        if n_expected_input == 0:
            raise ValueError("Number of Expected Input must be > 0")

        n_expected_gt: int = loader_json.get("expected_gt", 0)
        if n_expected_gt == 0:
            raise ValueError("Number of Expected Ground Truth must be > 0")

        if "image_ground_truth" in loader_json.keys():
            list_image_ground_truth_pairs: list = loader_json.get("image_ground_truth")

            # Go through each subject
            for subject_index, a_subject_image_ground_truth_pair in enumerate(
                list_image_ground_truth_pairs
            ):
                skip_subject_flag = False

                # 2 lists: Subject List + Ground Truth List
                assert len(a_subject_image_ground_truth_pair) == 2

                # Validate and trim Subject List
                list_subject_specific_images: list = a_subject_image_ground_truth_pair[
                    0
                ]
                if len(list_subject_specific_images) > n_expected_input:
                    list_subject_specific_images = list_subject_specific_images[
                        :n_expected_input
                    ]

                # Validate and trim Ground Truth List
                list_subject_specific_gt: list = a_subject_image_ground_truth_pair[1]
                if len(list_subject_specific_gt) > n_expected_gt:
                    list_subject_specific_gt = list_subject_specific_gt[:n_expected_gt]

                # Go check every file and if any of them don't exist, skip the subject
                for subject_images_index, path_file in enumerate(
                    list_subject_specific_images
                ):
                    list_subject_specific_images[subject_images_index] = ensure_absolute_path(
                        path_file, path_data
                    )
                    if not list_subject_specific_images[subject_images_index]:
                        skip_subject_flag = True

                # Exclude current subject
                if skip_subject_flag:
                    list_image_ground_truth_pairs[subject_index] = None
                    continue

                for gt_index, path_file in enumerate(list_subject_specific_gt):
                    list_subject_specific_gt[gt_index] = ensure_absolute_path(
                        path_file, path_data
                    )
                    if not list_subject_specific_images[gt_index]:
                        skip_subject_flag = True

                # Exclude current subject
                if skip_subject_flag:
                    list_image_ground_truth_pairs[subject_index] = None
                    continue

                # At this point, established ALL subject's related image and ground truth file exists.
                self.filename_pairs.append(
                    (
                        list_subject_specific_images,
                        list_subject_specific_gt,
                        None,  # No ROI for this dataset
                        None,  # No metadata for this dataset
                    )
                )

        return example_json


def ensure_absolute_path(path_potential_relative: Path, path_absolute_root: Path) -> str or None:
    """
    Check if a path is valid. If not, return the path combined with a predefined root path
    Args:
        path_potential_relative:
        path_absolute_root:

    Returns:

    """
    # If original path exists, then return its string version
    if Path(path_potential_relative).absolute().exists():
        return str(path_potential_relative)

    # If original path does not exist, then append it to the "common" data path and check if that exists
    # If that file exists, then return its string version
    elif (Path(path_absolute_root) / Path(path_potential_relative)).absolute().exists():
        return str(Path(path_absolute_root) / Path(path_potential_relative))
    else:
        return None

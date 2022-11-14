from __future__ import annotations
from pathlib import Path
import typing
from typing import List, Tuple
from loguru import logger
from ivadomed.loader.utils import ensure_absolute_path

if typing.TYPE_CHECKING:
    from ivadomed.loader.generalized_loader_configuration import (
        GeneralizedLoaderConfiguration,
    )

from ivadomed.loader.mri2d_segmentation_dataset import MRI2DSegmentationDataset

from ivadomed.keywords import (
    ModelParamsKW,
    MetadataKW, DataloaderKW, FileMissingHandleKW,
)


class FilesDataset(MRI2DSegmentationDataset):
    """Files Sets specific dataset loader"""

    def __init__(
        self, dict_files_pairing: dict, config: GeneralizedLoaderConfiguration
    ):
        """
        Constructor that leverage a generalized loader configuration
        Args:
            dict_files_pairing: An example shown above in example_json dict.
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

        if dict_files_pairing.get(DataloaderKW.MISSING_FILES_HANDLE) == FileMissingHandleKW.SKIP:
            self.drop_missing = True

        self.n_expected_input: int = int(dict_files_pairing.get(DataloaderKW.EXPECTED_INPUT, 0))
        if self.n_expected_input <= 0:
            raise ValueError("Number of Expected Input at File Dataset level must be > 0")

        self.n_expected_gt: int = int(dict_files_pairing.get(DataloaderKW.EXPECTED_GT, 0))
        if self.n_expected_gt <= 0:
            raise ValueError("Number of Expected Ground Truth at File Dataset level must be > 0")

        self.name: str = dict_files_pairing.get(DataloaderKW.SUBSET_LABEL, "DefaultFileDatasetLabel")

        # Some key assumptions for this implementation going forard:
        # 1. We assume user explicitly provide the subject lists so WE do not do any additional filtering

        # Currently does not support contrast balance (See BIDS Data 2D for reference implementation)
        # Create a dictionary with the number of subjects for each contrast of contrast_balance

        # NOT going to support bounding boxes

        # NOT going to care about multi-contrast/channels as we assume user explicitly provide that.

        # Get all derivatives filenames frm User JSON

        #################
        # Create filename_pairs
        #################
        self.path_data, self.filename_pairs = self.parse_spec_json_and_update_filename_pairs(dict_files_pairing)

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
            length=length,
            stride=stride,
            slice_axis=config.slice_axis,
            nibabel_cache=config.nibabel_cache,
            transform=config.transform,
            slice_filter_fn=config.slice_filter_fn,
            patch_filter_fn=config.patch_filter_fn,
            task=config.task,
            roi_params=config.roi_params,
            soft_gt=config.soft_gt,
            is_input_dropout=config.is_input_dropout,
            disk_cache=config.disk_cache,
        )

    def parse_spec_json_and_update_filename_pairs(self, loader_json: dict) -> Tuple[str, list]:
        """Load the json file and return the dictionary"""
        # Given a JSON file, try to load the file pairing from it.
        assert loader_json.get(DataloaderKW.TYPE) == "FILES"

        self.path_data = loader_json.get(DataloaderKW.PATH_DATA, ".")

        if DataloaderKW.IMAGE_GROUND_TRUTH in loader_json.keys():
            self.filename_pairs = self.validate_update_filename_pairs(
                loader_json,
                self.n_expected_gt,
                self.n_expected_input,
                self.path_data
            )

        return self.path_data, self.filename_pairs

    def validate_update_filename_pairs(self,
                                       loader_json: dict,
                                       n_expected_gt: int,
                                       n_expected_input: int,
                                       path_data: Path) -> list:
        """
        Validate the JSON file and update the filename_pairs after ensuring no extra or insufficient files presents.

        Args:
            loader_json:
            n_expected_gt:
            n_expected_input:
            path_data:

        Returns:

        """
        filename_pairs = []

        if not loader_json.get(DataloaderKW.IMAGE_GROUND_TRUTH):
            raise KeyError("Expected V2 loader configuration files but the Image/Groundtruth key was not found in the loader JSON.")

        list_image_ground_truth_pairs: list = loader_json.get(DataloaderKW.IMAGE_GROUND_TRUTH)
        list_image_ground_truth_pairs_filtered: list = []
        # Go through each subject
        for a_subject_image_ground_truth_pair in list_image_ground_truth_pairs:

            skip_subject_flag = False

            # 2 lists: Subject List + Ground Truth list
            assert len(a_subject_image_ground_truth_pair) == 2

            # Validate and trim Subject files list
            list_subject_specific_images: list = a_subject_image_ground_truth_pair[0]
            if len(list_subject_specific_images) > n_expected_input:
                list_subject_specific_images = list_subject_specific_images[:n_expected_input]

            # Validate and trim Ground Truth files list
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
                continue

            if len(list_subject_specific_images) < n_expected_input:
                logger.warning(f"Fewer input files found than expected for subject specification "
                               f"{a_subject_image_ground_truth_pair}. Expected {n_expected_input} "
                               f"but found {len(list_subject_specific_images)} from {list_subject_specific_images}")
                if self.drop_missing:
                    continue

            if len(list_subject_specific_gt) < n_expected_gt:
                logger.warning(f"Fewer ground truth files found than expected, for subject specification "
                               f"{a_subject_image_ground_truth_pair}. Expected {n_expected_gt} "
                               f"but found {len(list_subject_specific_gt)} from {list_subject_specific_gt}")
                if self.drop_missing:
                    continue

            for gt_index, path_file in enumerate(list_subject_specific_gt):
                list_subject_specific_gt[gt_index] = ensure_absolute_path(
                    path_file, path_data
                )
                if not list_subject_specific_images[gt_index]:
                    skip_subject_flag = True

            # Exclude current subject
            if skip_subject_flag:
                continue

            # Generate simple meta data #todo: should load json if present.
            metadata: List[dict] = [{}]* len(list_subject_specific_images) # "data_specificiation_type": "file_dataset"

            list_image_ground_truth_pairs_filtered.append((list_subject_specific_images, list_subject_specific_gt))

            # At this point, established ALL subject's related image and ground truth file exists.
            filename_pairs.append(
                (
                    list_subject_specific_images,
                    list_subject_specific_gt,
                    None,  # No ROI for this dataset, String?
                    metadata,  # No metadata for this dataset, Dict?
                )
            )

        loader_json[DataloaderKW.IMAGE_GROUND_TRUTH] = list_image_ground_truth_pairs_filtered

        return filename_pairs

    def preview(self, verbose: bool = False):
        """
        Preview the FINALIZED FileDataSet included that has undergone validation and is ready to be used for training.
        Args:
            verbose: whether to print out the actual data path
        """
        logger.info(f"\t\tFileDataset object from {self.path_data}, {len(self.filename_pairs)} pairs of data files")

        if verbose:
            for pair_index, a_pair in enumerate(self.filename_pairs):
                logger.info(f"\t\t\tImage Pair {pair_index}, Subject Image(s):")
                for a_image in a_pair[0]:
                    logger.info(f"\t\t\t\t{a_image}")
                logger.info(f"\t\t\tImage Pair {pair_index}, Ground Truth Image(s):")
                for a_gt in a_pair[1]:
                    logger.info(f"\t\t\t\t{a_gt}")

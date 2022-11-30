from __future__ import annotations
from pathlib import Path
import typing
from typing import List, Tuple
from loguru import logger

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





from __future__ import annotations
import typing
from typing import List, Tuple

from ivadomed.keywords import ModelParamsKW, DataloaderKW
from ivadomed.loader.files_dataset_group import FileDatasetGroup

if typing.TYPE_CHECKING:
    from ivadomed.loader.all_dataset_group import AllDatasetGroups
from ivadomed.loader.generalized_loader_configuration import GeneralizedLoaderConfiguration
from ivadomed.loader.mri2d_segmentation_dataset import MRI2DSegmentationDataset

from loguru import logger


class ConsolidatedDataset(MRI2DSegmentationDataset):
    """
    A consolidated dataset is a harmonized way to handle BIDSDataset and FilesDataset.
    The consolidated dataset is a list of tuples of the form:
    (list of input files, list of ground truth files, ROI filename, metadata)
    """

    def __init__(
        self, filename_pairs: list, config: GeneralizedLoaderConfiguration
    ):
        """Initialize the consolidated dataset.

        Args:
            config (GeneralizedLoaderConfiguration): The configuration.
        """
        # This is the key attribute that needs to be populated once data loading is complete.
        self.filename_pairs: List[Tuple[list, list, str, dict]] = filename_pairs
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
            config.disk_cache
        )

    @staticmethod
    def consolidate_AllDatasetGroups_to_a_specific_filedataset_type(
            all_dataset_groups: AllDatasetGroups,
            consolidation_type: str
    ) -> ConsolidatedDataset:
        """
        Consolidate data across "all the dataset groups" into a single ConsolidatedDataset object.

        Args:
            consolidation_type:
            all_dataset_groups (AllDatasetGroups): All the dataset groups.

        Returns:
            ConsolidatedDataset: A single dataset group.
        """

        filename_pairs: List[Tuple[list, list, str, dict]] = []

        if consolidation_type == DataloaderKW.TRAINING:
            for dataset_filename_pairs in all_dataset_groups.train_filename_pairs:
                filename_pairs.append(dataset_filename_pairs)
        elif consolidation_type == DataloaderKW.VALIDATION:
            for dataset_filename_pairs in all_dataset_groups.val_filename_pairs:
                filename_pairs.append(dataset_filename_pairs)
        elif consolidation_type == DataloaderKW.TEST:
            for dataset_filename_pairs in all_dataset_groups.test_filename_pairs:
                filename_pairs.append(dataset_filename_pairs)
        else:
            raise ValueError(f"Unknown consolidation type: {consolidation_type}")

        return ConsolidatedDataset(filename_pairs, all_dataset_groups.config)

    @staticmethod
    def consolidate_DatasetGroup_to_a_specific_filedataset_type(
            data_group: FileDatasetGroup,
            consolidation_type: str
    ) -> ConsolidatedDataset:
        """
        Consolidate data within a data_group under a ConsolidatedDataset object.

        Args:
            consolidation_type:
            data_group (FileDatasetGroup): A dataset group with many subsets of dataset.

        Returns:
            ConsolidatedDataset: A single dataset group.
        """

        filename_pairs: List[Tuple[list, list, str, dict]] = []

        if consolidation_type == DataloaderKW.TRAINING:
            for dataset_filename_pairs in data_group.train_filename_pairs:
                filename_pairs.append(dataset_filename_pairs)
        elif consolidation_type == DataloaderKW.VALIDATION:
            for dataset_filename_pairs in data_group.val_filename_pairs:
                filename_pairs.append(dataset_filename_pairs)
        elif consolidation_type == DataloaderKW.TEST:
            for dataset_filename_pairs in data_group.test_filename_pairs:
                filename_pairs.append(dataset_filename_pairs)
        else:
            raise ValueError(f"Unknown consolidation type: {consolidation_type}")

        return ConsolidatedDataset(filename_pairs, data_group.config)

    def preview(self, verbose: bool = False):
        """
        Preview the FINALIZED ConsolidatedDataset pairing included that has undergone validation and is ready to be
        used for training.

        Args:
            verbose: whether to print out the actual data path
        """
        logger.info(f"\t\tConsolidatedDataset object composed of {len(self.filename_pairs)} pairs of data files")

        if verbose:
            for pair_index, a_pair in enumerate(self.filename_pairs):
                logger.info(f"\t\t\tImage Pair {pair_index}, Subject Image(s):")
                for a_image in a_pair[0]:
                    logger.info(f"\t\t\t\t{a_image}")
                logger.info(f"\t\t\tImage Pair {pair_index}, Ground Truth Image(s):")
                for a_gt in a_pair[1]:
                    logger.info(f"\t\t\t\t{a_gt}")

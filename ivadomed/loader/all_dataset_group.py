from __future__ import annotations
import typing
from typing import List, Tuple
from ivadomed.keywords import DataloaderKW
from ivadomed.loader.files_dataset_group import FileDatasetGroup

if typing.TYPE_CHECKING:
    from ivadomed.loader.generalized_loader_configuration import (
        GeneralizedLoaderConfiguration,
    )
from loguru import logger


class AllDatasetGroups:
    # This class will encompass ALL DATA within a study.

    def __init__(self,
                 dict_all_datasets_group_spec: dict,
                 config: GeneralizedLoaderConfiguration
                 ):
        """
        This contains all study related DatasetGroups and each FileDatasetGroup contain its own FileDataset
        Args:
            dict_all_datasets_group_spec: An example shown above in example_json dict.
            config:
        """
        self.list_dataset_groups: List[FileDatasetGroup] = []
        self.list_dataset_groups_names: List[str] = []

        # These are lists of either Files3DDataset or FilesDataset, aggregated across all the DatasetGroups within.
        self.train_flat_fileset = []
        self.val_flat_fileset = []
        self.test_flat_fileset = []

        # These are lists of underlying pairing
        self.train_filename_pairs: List[Tuple[list, list, str, dict]] = []
        self.val_filename_pairs: List[Tuple[list, list, str, dict]] = []
        self.test_filename_pairs: List[Tuple[list, list, str, dict]] = []

        # Store the Generalized configuration
        self.config = config

        # Store the generalized expected number of files which will be enforced.
        self.n_expected_input: int = int(dict_all_datasets_group_spec.get(DataloaderKW.EXPECTED_INPUT, 0))
        if self.n_expected_input <= 0:
            raise ValueError("At AllDataSetGroup level, Number of Expected Input must be > 0")

        self.n_expected_gt: int = int(dict_all_datasets_group_spec.get(DataloaderKW.EXPECTED_GT, 0))
        if self.n_expected_gt <= 0:
            raise ValueError("At AllDataSetGroup level, Number of Expected Ground Truth must be > 0")

        # Instantiate each the DatasetGroups
        for dict_dataset_group_spec in dict_all_datasets_group_spec.get(DataloaderKW.DATASET_GROUPS):

            # Create a new Dataset Group
            dataset_group = FileDatasetGroup(dict_dataset_group_spec, config)

            # Conduct skip check for aggregation
            skip = False
            if dataset_group.n_expected_input != self.n_expected_input:
                logger.warning(f"Skipping handling of FileDatasetGroup {dataset_group.name} as its Number of Expected Input in the "
                               f"{dataset_group.n_expected_input} does not match the number of "
                               f"Expected Input in the AllDatasetGroup {self.n_expected_input}")
                skip = True
            if dataset_group.n_expected_gt != self.n_expected_gt:
                logger.warning(f"Skipping handling of FileDatasetGroup {dataset_group.name} as its Number of Expected Ground Truth in the "
                               f"{dataset_group.n_expected_gt} does not match the number of "
                               f"Expected Ground Truth in the Train FileDatasetGroup {self.n_expected_gt}")
                skip = True

            if skip:
                continue

            # Track the dataset group created.
            self.list_dataset_groups.append(dataset_group)

            # These are lists of underlying pairing
            self.train_filename_pairs.extend(dataset_group.train_filename_pairs)
            self.val_filename_pairs.extend(dataset_group.val_filename_pairs)
            self.test_filename_pairs.extend(dataset_group.test_filename_pairs)

            # Append the name of the dataset label
            self.list_dataset_groups_names.append(
                dataset_group.name
            )

        # Validate that at least SOME train/val/test pairs are specified
        if not self.train_filename_pairs:
            raise ValueError("No training data specified. Please ensure at least one dataset group has training data.")
        if not self.val_filename_pairs:
            raise ValueError("No validation data specified. Please ensure at least one dataset group has validation data.")
        if not self.test_filename_pairs:
            raise ValueError("No testing data specified. Please ensure at least one dataset group has testing data.")


    def validate_IO_across_datagroups(self):
        """
        Future stub to use to either clean up excessive files dataset groups or drop insufficient number
        Returns:

        """
        pass

    def preview(self, verbose=False):
        """
        Preview the FINALIZED AllDataSetGroupS included that has undergone validation and is ready to be used for training.
        Args:
            verbose: whether to print out the actual data path
        """
        all_groups_name = f"AllDatasetGroups({self.list_dataset_groups_names})"
        logger.info(all_groups_name)
        for dataset_group in self.list_dataset_groups:
            dataset_group.preview(verbose)


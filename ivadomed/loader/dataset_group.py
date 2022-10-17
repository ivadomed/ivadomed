from __future__ import annotations
import typing
from typing import List, Tuple
from ivadomed.loader.files3d_dataset import Files3DDataset
from ivadomed.loader.files_dataset import FilesDataset
from loguru import logger
if typing.TYPE_CHECKING:
    from ivadomed.loader.generalized_loader_configuration import (
        GeneralizedLoaderConfiguration,
    )

from ivadomed.keywords import (
    ModelParamsKW,
    DataloaderKW,
)


class DatasetGroup:
    # Each file group must have training/validation/test keys.

    def __init__(self,
                 dict_file_group_spec: dict,
                 config: GeneralizedLoaderConfiguration
                 ):
        """
        Constructor that leverage a generalized loader configuration
        Args:
            dict_file_group_spec: An example shown above in example_json dict.
            config:
        """
        # These are lists of either Files3DDataset or FilesDataset
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []

        # These are lists of underlying pairing
        self.train_filename_pairs: List[Tuple[list, list, str, dict]] = []
        self.val_filename_pairs: List[Tuple[list, list, str, dict]] = []
        self.test_filename_pairs: List[Tuple[list, list, str, dict]] = []

        # DatasetGroup config
        self.config = config

        # This sets the expected number of files at the dataset group level. Useful for validating subsequent filesets and whether they are compliant or not.
        self.n_expected_input: int = int(dict_file_group_spec.get(DataloaderKW.EXPECTED_INPUT, 0))
        if self.n_expected_input <= 0:
            raise ValueError("At DataSetGroup level, Number of Expected Input must be > 0")

        self.n_expected_gt: int = int(dict_file_group_spec.get(DataloaderKW.EXPECTED_GT, 0))
        if self.n_expected_gt <= 0:
            raise ValueError("At DataSetGroup level, Number of Expected Ground Truth must be > 0")

        # This is the key attribute that needs to be populated once data loading is complete.
        if (ModelParamsKW.IS_2D not in config.model_params and not config.model_params.get(ModelParamsKW.IS_2D)):
            self.validate_update_3d_train_val_test_and_filename_pairs(config, dict_file_group_spec)
        else:
            self.validate_update_2d_train_val_test_dataset_and_filename_pairs(config, dict_file_group_spec)

        # Assign default labels if not provided.
        self.name: str = dict_file_group_spec.get(DataloaderKW.DATASET_GROUP_LABEL, "DefaultDatasetGroupLabel")

    def validate_update_3d_train_val_test_and_filename_pairs(self, config, dict_file_group_spec):
        # Instantiate all the 3D Train/Val/Test Datasets
        for a_train_dataset in dict_file_group_spec[DataloaderKW.TRAINING]:
            if a_train_dataset[DataloaderKW.TYPE] == "FILES":
                train_set = Files3DDataset(a_train_dataset, config)
                self.train_dataset.append(train_set)
                self.train_filename_pairs.extend(train_set.filename_pairs)
        for a_val_dataset in dict_file_group_spec[DataloaderKW.VALIDATION]:
            if a_val_dataset[DataloaderKW.TYPE] == "FILES":
                val_set = Files3DDataset(a_val_dataset, config)
                self.val_dataset.append(val_set)
                self.val_filename_pairs.extend(val_set.filename_pairs)
        for a_test_dataset in dict_file_group_spec[DataloaderKW.TEST]:
            if a_test_dataset[DataloaderKW.TYPE] == "FILES":
                test_set = Files3DDataset(a_test_dataset, config)
                self.test_dataset.append(test_set)
                self.test_filename_pairs.extend(test_set.filename_pairs)

    def validate_update_2d_train_val_test_dataset_and_filename_pairs(self, config, dict_file_group_spec):
        # Instantiate all the 2D Train/Val/Test Datasets
        for a_train_dataset in dict_file_group_spec[DataloaderKW.TRAINING]:

            if a_train_dataset[DataloaderKW.TYPE] == "FILES":

                train_set = FilesDataset(a_train_dataset, config)

                if train_set.n_expected_input == self.n_expected_input and train_set.n_expected_gt == self.n_expected_gt:
                    self.train_dataset.append(train_set)
                    self.train_filename_pairs.extend(train_set.filename_pairs)
                if train_set.n_expected_input != self.n_expected_input:
                    logger.warning(f"Skipping handling of {train_set.name} as its Number of Expected Input in the "
                                   f"Train Dataset {train_set.n_expected_input} does not match the number of "
                                   f"Expected Input in the Train DatasetGroup {self.n_expected_input}")
                if train_set.n_expected_gt != self.n_expected_gt:
                    logger.warning(f"Skipping handling of {train_set.name} as its Number of Expected Ground Truth in the "
                                   f"Train Dataset {train_set.n_expected_gt} does not match the number of "
                                   f"Expected Ground Truth in the Train DatasetGroup {self.n_expected_gt}")

        for a_val_dataset in dict_file_group_spec[DataloaderKW.VALIDATION]:

            if a_val_dataset[DataloaderKW.TYPE] == "FILES":
                val_set = FilesDataset(a_val_dataset, config)

                if val_set.n_expected_input == self.n_expected_input and val_set.n_expected_gt == self.n_expected_gt:
                    self.val_dataset.append(val_set)
                    self.val_filename_pairs.extend(val_set.filename_pairs)
                if val_set.n_expected_input != self.n_expected_input:
                    logger.warning(f"Skipping handling of {val_set.name} as its Number of Expected Input in the "
                                   f"Validation Dataset {val_set.n_expected_input} does not match the number of "
                                   f"Expected Input in the Validation DatasetGroup {self.n_expected_input}")
                if val_set.n_expected_gt != self.n_expected_gt:
                    logger.warning(f"Skipping handling of {val_set.name} as its Number of Expected Ground Truth in the "
                                   f"Validation Dataset {val_set.n_expected_gt} does not match the number of "
                                   f"Expected Ground Truth in the Validation DatasetGroup {self.n_expected_gt}")

        for a_test_dataset in dict_file_group_spec[DataloaderKW.TEST]:

            if a_test_dataset[DataloaderKW.TYPE] == "FILES":
                test_set = FilesDataset(a_test_dataset, config)

                if test_set.n_expected_input == self.n_expected_input and test_set.n_expected_gt == self.n_expected_gt:
                    self.test_dataset.append(test_set)
                    self.test_filename_pairs.extend(test_set.filename_pairs)
                if test_set.n_expected_input != self.n_expected_input:
                    logger.warning(f"Skipping handling of {test_set.name} as its Number of Expected Input in the "
                                   f"Test Dataset {test_set.n_expected_input} does not match the number of "
                                   f"Expected Input in the Test DatasetGroup {self.n_expected_input}")
                if test_set.n_expected_gt != self.n_expected_gt:
                    logger.warning(f"Skipping handling of {test_set.name} as its Number of Expected Ground Truth in the "
                                   f"Test Dataset {test_set.n_expected_gt} does not match the number of "
                                   f"Expected Ground Truth in the Test DatasetGroup {self.n_expected_gt}")


    def preview(self, verbose=False):
        """
        Preview the FINALIZED DataSetGroup included that has undergone validation and is ready to be used for training.
        Args:
            verbose: whether to print out the actual data path
        """
        logger.info(f"DatasetGroup: {self.name}")

        logger.info("\tTrain:")
        for train_set in self.train_dataset:
            train_set.preview(verbose)

        logger.info("\tVal:")
        for val_set in self.val_dataset:
            val_set.preview(verbose)

        logger.info("\tTest:")
        for test_set in self.test_dataset:
            test_set.preview(verbose)


    def validate_IO_across_datasets(self):
        pass
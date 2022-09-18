from __future__ import annotations
from pathlib import Path
import typing
from typing import List, Tuple
from pprint import pformat, pprint
from ivadomed.loader.files3d_dataset import Files3DDataset
from ivadomed.loader.files_dataset import FilesDataset
from ivadomed.loader.utils import ensure_absolute_path
from loguru import logger
if typing.TYPE_CHECKING:
    from ivadomed.loader.generalized_loader_configuration import (
        GeneralizedLoaderConfiguration,
    )

from ivadomed.loader.mri2d_segmentation_dataset import MRI2DSegmentationDataset

from ivadomed.keywords import (
    ModelParamsKW,
    MetadataKW, DataloaderKW,
)

example_DatasetGroup_json: dict = {
    "datasetgroup_label": "DataSet1",
    "training": [
        {
            "type": "FILES",
            "subset_label": "Subset3",
            "image_ground_truth": [
                [["/Path to/subset_folder_2/sub1_T1.nii", "/Path to/subset_folder_2/sub1_T2.nii"],
                 ["/Path to/subset_folder_2/gt1"]],
                [["/Path to/subset_folder_2/sub2_T1.nii", "/Path to/subset_folder_2/sub1_T1.nii"],
                 ["/Path to/subset_folder_2/gt2"]],
                [["/Path to/subset_folder_2/sub3_T1.nii", "/Path to/subset_folder_2/sub1_T1.nii"],
                 ["/Path to/subset_folder_2/gt3"]]
            ],
            "expected_input": 2,
            "expected_gt": 1,
            "missing_files_handle": "drop_subject",
            "excessive_files_handle": "use_first_and_warn"
            # "path_data": no path data key as absolute path required for image_ground_truth pairing
        }
    ],
    "validation": "[ Same Above Dict struct]",
    "test": "[Same Above Dict struct]",
}


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

        # This is the key attribute that needs to be populated once data loading is complete.
        if (ModelParamsKW.IS_2D in config.model_params and not config.model_params.get(ModelParamsKW.IS_2D)):
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
        else:
            # Instantiate all the 2D Train/Val/Test Datasets
            for a_train_dataset in dict_file_group_spec[DataloaderKW.TRAINING]:
                if a_train_dataset[DataloaderKW.TYPE] == "FILES":
                    train_set = FilesDataset(a_train_dataset, config)
                    self.train_dataset.append(train_set)
                    self.train_filename_pairs.extend(train_set.filename_pairs)
            for a_val_dataset in dict_file_group_spec[DataloaderKW.VALIDATION]:
                if a_val_dataset[DataloaderKW.TYPE] == "FILES":
                    val_set = FilesDataset(a_val_dataset, config)
                    self.val_dataset.append(val_set)
                    self.val_filename_pairs.extend(val_set.filename_pairs)
            for a_test_dataset in dict_file_group_spec[DataloaderKW.TEST]:
                if a_test_dataset[DataloaderKW.TYPE] == "FILES":
                    test_set = FilesDataset(a_test_dataset, config)
                    self.test_dataset.append(test_set)
                    self.test_filename_pairs.extend(test_set.filename_pairs)

        # Assign default labels if not provided.
        self.filegroup_label: str = dict_file_group_spec.get(DataloaderKW.DATASET_GROUP_LABEL, "DefaultFileGroupLabel")

    def preview(self, verbose=False):
        """
        Preview the FINALIZED DataSetGroup included that has undergone validation and is ready to be used for training.
        Args:
            verbose: whether to print out the actual data path
        """
        logger.info(f"DatasetGroup: {self.filegroup_label}")

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
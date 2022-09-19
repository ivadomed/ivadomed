from __future__ import annotations
import typing
from typing import List, Tuple
from pprint import pformat, pprint
from ivadomed.keywords import DataloaderKW
from ivadomed.loader.dataset_group import DatasetGroup

if typing.TYPE_CHECKING:
    from ivadomed.loader.generalized_loader_configuration import (
        GeneralizedLoaderConfiguration,
    )
from loguru import logger


example_all_dataset_groups_config_json: dict = {
    "dataset_groups": [
        {
            "dataset_group_label": "DataSetGroup1",
            "training": [
                {
                    "type": "FILES",
                    "subset_label": "TrainFileDataSet1",
                    "image_ground_truth": [
                        [["sub-01/ses-01/anat/sub-01_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-01/ses-01/anat/sub-01_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-01/ses-01/anat/sub-01_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-01/ses-02/anat/sub-01_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-01/ses-02/anat/sub-01_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-01/ses-02/anat/sub-01_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-01/ses-03/anat/sub-01_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-01/ses-03/anat/sub-01_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-01/ses-03/anat/sub-01_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-01/ses-03/anat/sub-01_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-01/ses-04/anat/sub-01_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-01/ses-04/anat/sub-01_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 2,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",
                },
            ],
            "validation": [
                {
                    "type": "FILES",
                    "subset_label": "ValFileDataSet1",
                    "image_ground_truth": [
                        [["sub-02/ses-01/anat/sub-02_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-02/ses-01/anat/sub-02_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-02/ses-01/anat/sub-02_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-02/ses-02/anat/sub-02_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-02/ses-02/anat/sub-02_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-02/ses-02/anat/sub-02_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-02/ses-03/anat/sub-02_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-02/ses-03/anat/sub-02_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-02/ses-03/anat/sub-02_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-02/ses-03/anat/sub-02_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-02/ses-04/anat/sub-02_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-02/ses-04/anat/sub-02_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 2,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",

                }
            ],
            "test": [
                {
                    "type": "FILES",
                    "subset_label": "TestFileDataSet1",
                    "image_ground_truth": [
                        [["sub-03/ses-01/anat/sub-03_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-03/ses-01/anat/sub-03_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-03/ses-01/anat/sub-03_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-03/ses-02/anat/sub-03_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-03/ses-02/anat/sub-03_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-03/ses-02/anat/sub-03_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-03/ses-03/anat/sub-03_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-03/ses-03/anat/sub-03_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-03/ses-03/anat/sub-03_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-03/ses-03/anat/sub-03_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-03/ses-04/anat/sub-03_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-03/ses-04/anat/sub-03_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 2,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",
                }
            ],

        },
        {
            "dataset_group_label": "DataSetGroup2",
            "training": [
                {
                    "type": "FILES",
                    "subset_label": "TrainFileDataSet1",
                    "image_ground_truth": [
                        [["sub-04/ses-01/anat/sub-04_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-04/ses-01/anat/sub-04_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-04/ses-01/anat/sub-04_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-04/ses-02/anat/sub-04_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-04/ses-02/anat/sub-04_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-04/ses-02/anat/sub-04_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-04/ses-03/anat/sub-04_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-04/ses-03/anat/sub-04_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-04/ses-03/anat/sub-04_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-04/ses-03/anat/sub-04_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-04/ses-04/anat/sub-04_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-04/ses-04/anat/sub-04_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 2,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",
                },
            ],
            "validation": [
                {
                    "type": "FILES",
                    "subset_label": "ValFileDataSet1",
                    "image_ground_truth": [
                        [["sub-05/ses-01/anat/sub-05_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-05/ses-01/anat/sub-05_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-05/ses-01/anat/sub-05_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-05/ses-02/anat/sub-05_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-05/ses-02/anat/sub-05_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-05/ses-02/anat/sub-05_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-05/ses-03/anat/sub-05_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-05/ses-03/anat/sub-05_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-05/ses-03/anat/sub-05_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-05/ses-03/anat/sub-05_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-05/ses-04/anat/sub-05_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-05/ses-04/anat/sub-05_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 2,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",
                }
            ],
            "test": [
                {
                    "type": "FILES",
                    "subset_label": "TestFileDataSet1",
                    "image_ground_truth": [
                        [["sub-06/ses-01/anat/sub-06_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-06/ses-01/anat/sub-06_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-06/ses-01/anat/sub-06_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-06/ses-02/anat/sub-06_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-06/ses-02/anat/sub-06_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-06/ses-02/anat/sub-06_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-06/ses-03/anat/sub-06_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-06/ses-03/anat/sub-06_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-06/ses-03/anat/sub-06_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-06/ses-03/anat/sub-06_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-06/ses-04/anat/sub-06_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-06/ses-04/anat/sub-06_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    "expected_input": 2,
                    "expected_gt": 1,
                    "missing_files_handle": "drop_subject",
                    "excessive_files_handle": "use_first_and_warn",
                    "path_data": r"C:\Temp\Test",
                }
            ],

        },
    ]
}

class AllDatasetGroups:
    # This class will encompass ALL DATA within a study.

    def __init__(self,
                 dict_all_datasets_group_spec: dict,
                 config: GeneralizedLoaderConfiguration
                 ):
        """
        This contains all study related DatasetGroups and each DatasetGroup contain its own FileDataset
        Args:
            dict_all_datasets_group_spec: An example shown above in example_json dict.
            config:
        """
        self.list_dataset_groups: List[DatasetGroup] = []
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

        # Instantiate each the DatasetGroups
        for dict_dataset_group_spec in dict_all_datasets_group_spec.get(DataloaderKW.DATASET_GROUPS):

            # Create a new Dataset Group
            dataset_group = DatasetGroup(dict_dataset_group_spec, config)

            # Track the dataset group created.
            self.list_dataset_groups.append(dataset_group)

            # These are lists of either Files3DDataset or FilesDataset, aggregated across all the DatasetGroups within.
            self.train_flat_fileset.append(dataset_group.train_dataset)
            self.val_flat_fileset.append(dataset_group.val_dataset)
            self.test_flat_fileset.append(dataset_group.test_dataset)

            # These are lists of underlying pairing
            self.train_filename_pairs.extend(dataset_group.train_filename_pairs)
            self.val_filename_pairs.extend(dataset_group.val_filename_pairs)
            self.test_filename_pairs.extend(dataset_group.test_filename_pairs)

            # Append the name of the dataset label
            self.list_dataset_groups_names.append(
                dataset_group.filegroup_label
            )

    def validate_IO_across_datagroups(self):
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


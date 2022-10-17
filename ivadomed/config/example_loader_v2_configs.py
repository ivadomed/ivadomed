from __future__ import annotations

from ivadomed.keywords import DataloaderKW, FileMissingHandleKW, FileExcessiveHandleKW

"""
This is a centralized place to store all the example config files for the loader.
Purpose is to provide an updated view of how the dictionary structures can be used to reflect latest examples

"""

example_FileDataset_json: dict = {
    DataloaderKW.TYPE: "FILES",
    DataloaderKW.SUBSET_LABEL: "Subset3",
    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
    DataloaderKW.EXPECTED_INPUT: 2,
    DataloaderKW.EXPECTED_GT: 1,
    "meta_data_csv": "/metadata.csv",  # assumed to be the same shape as the default run.
    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
    "path_data": r"C:/Temp/Test",
}

example_DatasetGroup_json: dict = {
    DataloaderKW.DATASET_GROUP_LABEL: "DataSet1",
    DataloaderKW.TRAINING: [
        {
            DataloaderKW.TYPE: "FILES",
            DataloaderKW.SUBSET_LABEL: "Subset3",
            DataloaderKW.IMAGE_GROUND_TRUTH: [
                [["/Path to/subset_folder_2/sub1_T1.nii", "/Path to/subset_folder_2/sub1_T2.nii"],
                 ["/Path to/subset_folder_2/gt1"]],
                [["/Path to/subset_folder_2/sub2_T1.nii", "/Path to/subset_folder_2/sub1_T1.nii"],
                 ["/Path to/subset_folder_2/gt2"]],
                [["/Path to/subset_folder_2/sub3_T1.nii", "/Path to/subset_folder_2/sub1_T1.nii"],
                 ["/Path to/subset_folder_2/gt3"]]
            ],
            DataloaderKW.EXPECTED_INPUT: 2,
            DataloaderKW.EXPECTED_GT: 1,
            DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
            DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN
            # "path_data": no path data key as absolute path required for image_ground_truth pairing
        }
    ],
    DataloaderKW.VALIDATION: "[ Same Above Dict struct]",
    DataloaderKW.TEST: "[Same Above Dict struct]",
    DataloaderKW.EXPECTED_INPUT: 2,
    DataloaderKW.EXPECTED_GT: 1,
}

example_all_dataset_groups_config_json: dict = {
    DataloaderKW.DATASET_GROUPS: [
        {
            DataloaderKW.DATASET_GROUP_LABEL: "DataSetGroup1",
            DataloaderKW.TRAINING: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "TrainFileDataSet1",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 2,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                },
            ],
            DataloaderKW.VALIDATION: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "ValFileDataSet1",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 2,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",

                }
            ],
            DataloaderKW.TEST: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "TestFileDataSet1",
                    DataloaderKW.IMAGE_GROUND_TRUTH:  [
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
                    DataloaderKW.EXPECTED_INPUT: 2,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                }
            ],
            DataloaderKW.EXPECTED_INPUT: 2,
            DataloaderKW.EXPECTED_GT: 1,
        },
        {
            DataloaderKW.DATASET_GROUP_LABEL: "DataSetGroup2",
            DataloaderKW.TRAINING: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "TrainFileDataSet1",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 2,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                },
            ],
            DataloaderKW.VALIDATION: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "ValFileDataSet1",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 2,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                }
            ],
            DataloaderKW.TEST: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "TestFileDataSet1",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 2,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                }
            ],
            DataloaderKW.EXPECTED_INPUT: 2,
            DataloaderKW.EXPECTED_GT: 1,
        },
    ],
    DataloaderKW.EXPECTED_INPUT: 2,
    DataloaderKW.EXPECTED_GT: 1,
}

example_uni_channel_all_dataset_groups_config_json: dict = {
    DataloaderKW.DATASET_GROUPS: [
        {
            DataloaderKW.DATASET_GROUP_LABEL: "DataSetGroup1",
            DataloaderKW.TRAINING: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "TrainFileDataSet1",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 1,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: "drop_subject",
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: "use_first_and_warn",
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                },
            ],
            DataloaderKW.VALIDATION: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "ValFileDataSet1",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 1,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: "drop_subject",
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: "use_first_and_warn",
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                }
            ],
            DataloaderKW.TEST: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "TestFileDataSet1",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 1,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: "drop_subject",
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: "use_first_and_warn",
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                }
            ],
            DataloaderKW.EXPECTED_INPUT: 1,
            DataloaderKW.EXPECTED_GT: 1,
        },
        {
            DataloaderKW.DATASET_GROUP_LABEL: "DataSetGroup2",
            DataloaderKW.TRAINING: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "TrainFileDataSet2",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 1,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: "drop_subject",
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: "use_first_and_warn",
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                },
            ],
            DataloaderKW.VALIDATION: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "ValFileDataSet2",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 1,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: "drop_subject",
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: "use_first_and_warn",
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                }
            ],
            DataloaderKW.TEST: [
                {
                    DataloaderKW.TYPE: "FILES",
                    DataloaderKW.SUBSET_LABEL: "TestFileDataSet2",
                    DataloaderKW.IMAGE_GROUND_TRUTH: [
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
                    DataloaderKW.EXPECTED_INPUT: 1,
                    DataloaderKW.EXPECTED_GT: 1,
                    DataloaderKW.MISSING_FILES_HANDLE: "drop_subject",
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: "use_first_and_warn",
                    DataloaderKW.PATH_DATA: r"C:\Temp\Test",
                }
            ],
            DataloaderKW.EXPECTED_INPUT: 1,
            DataloaderKW.EXPECTED_GT: 1,
        },
    ],
    DataloaderKW.EXPECTED_INPUT: 1,
    DataloaderKW.EXPECTED_GT: 1,
}

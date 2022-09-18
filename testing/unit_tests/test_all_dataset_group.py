from ivadomed.loader.all_dataset_group import AllDatasetGroups
from ivadomed.loader.dataset_group import DatasetGroup

from testing.common_testing_util import remove_tmp_dir
from testing.unit_tests.t_utils import (
    create_tmp_dir
)
from ivadomed.loader.generalized_loader_configuration import (
    GeneralizedLoaderConfiguration,
)


def setup_function():
    create_tmp_dir()


all_dataset_groups_config_json: dict = {
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


def test_all_dataset_group():
    model_dict = {
        "name": "Unet",
        "dropout_rate": 0.3,
        "bn_momentum": 0.1,
        "final_activation": "sigmoid",
        "depth": 3,
    }

    # Build a GeneralizedLoaderConfiguration:
    loader_config: GeneralizedLoaderConfiguration = GeneralizedLoaderConfiguration(
        model_params=model_dict,
    )
    a = AllDatasetGroups(all_dataset_groups_config_json, loader_config)
    a.preview(verbose=True)


def teardown_function():
    remove_tmp_dir()

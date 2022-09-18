from ivadomed.loader.files_dataset import FilesDataset
from testing.unit_tests.t_utils import (
    create_tmp_dir
)
from ivadomed.loader.generalized_loader_configuration import (
    GeneralizedLoaderConfiguration,
)


def setup_function():
    create_tmp_dir()


files_config_json: dict = {
    "type": "FILES",
    "subset_label": "Subset3",
    "image_ground_truth": [
        [
            [
                r"sub-02\ses-04\anat\sub-02_ses-04_acq-MToff_MTS.nii",
                r"sub-02\ses-04\anat\sub-02_ses-04_acq-MTon_MTS.nii",
            ],
            [
                r"derivatives\labels\sub-02\ses-04\anat\sub-02_ses-04_mt-off_MTS_lesion-manual-rater1.nii"
            ],
        ],
        [
            [
                r"sub-02\ses-05\anat\sub-02_ses-05_acq-MToff_MTS.nii",
                r"sub-02\ses-05\anat\sub-02_ses-05_acq-MTon_MTS.nii",
            ],
            [
                r"derivatives\labels\sub-02\ses-05\anat\sub-02_ses-05_mt-off_MTS_lesion-manual-rater1.nii"
            ],
        ],
        [
            [
                r"sub-02\ses-04\anat\sub-02_ses-04_flip-1_mt-off_MTS.nii",
                r"sub-02\ses-04\anat\sub-02_ses-04_flip-1_mt-on_MTS.nii",
            ],
            [
                r"derivatives\labels\sub-02\ses-04\anat\sub-02_ses-04_mt-off_MTS_lesion-manual-rater1.nii"
            ],
        ],
        [
            [
                r"sub-02\ses-04\anat\sub-02_ses-04_flip-1_mt-off_MTS.nii",
                r"sub-02\ses-04\anat\sub-02_ses-04_flip-1_mt-on_MTS.nii",
                r"sub-02\ses-04\anat\sub-02_ses-04_flip-2_mt-off_MTS.nii",
                r"sub-02\ses-04\anat\sub-02_ses-04_flip-2_mt-on_MTS.nii",
            ],
            [
                r"derivatives\labels\sub-02\ses-04\anat\sub-02_ses-04_mt-off_MTS_lesion-manual-rater1.nii"
            ],
        ],
    ],
    "expected_input": 2,
    "expected_gt": 1,
    "meta_data_csv": "metadata.csv",  # assumed to be the same shape as the default run.
    "missing_files_handle": "drop_subject",
    "excessive_files_handle": "use_first_and_warn",
    "path_data": r"C:\Temp\Test",
}


def test_FilesDataset():
    model_dict = {
        "name": "Unet",
        "dropout_rate": 0.3,
        "bn_momentum": 0.1,
        "final_activation": "sigmoid",
        "depth": 3,
    }

    # Build a GeneralizedLoaderConfiguration:
    model_config_json: GeneralizedLoaderConfiguration = GeneralizedLoaderConfiguration(
        model_params=model_dict,
    )
    a = FilesDataset(files_config_json, model_config_json)
    a.preview()

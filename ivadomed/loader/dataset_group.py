from __future__ import annotations
import typing
from typing import List, Tuple
from ivadomed.loader.files3d_dataset import Files3DDataset
from ivadomed.loader.files_dataset import FilesDataset
from loguru import logger

from ivadomed.loader.utils import ensure_absolute_path

if typing.TYPE_CHECKING:
    from ivadomed.loader.generalized_loader_configuration import (
        GeneralizedLoaderConfiguration,
    )

from ivadomed.keywords import (
    ModelParamsKW,
    DataloaderKW, FileMissingHandleKW,
)


class FileDatasetGroup:
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
        print(dict_file_group_spec)

        # These are lists of underlying pairing
        self.train_filename_pairs: List[Tuple[list, list, str, dict]] = []
        self.val_filename_pairs: List[Tuple[list, list, str, dict]] = []
        self.test_filename_pairs: List[Tuple[list, list, str, dict]] = []


        # FileDatasetGroup config
        self.config = config

        # The common path shared by all their
        self.path_data = dict_file_group_spec.get(DataloaderKW.PATH_DATA, ".")

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
            self.validate_update_2d_train_val_test_dataset_and_filename_pairs(dict_file_group_spec)

        # Assign default labels if not provided.
        self.name: str = dict_file_group_spec.get(DataloaderKW.DATASET_GROUP_LABEL, "DefaultDatasetGroupLabel")

    def validate_update_3d_train_val_test_and_filename_pairs(self, config, dict_file_group_spec):
        # Instantiate all the 3D Train/Val/Test Datasets
        for a_train_dataset in dict_file_group_spec[DataloaderKW.TRAINING]:
            if a_train_dataset[DataloaderKW.TYPE] == "FILES":
                train_set = Files3DDataset(a_train_dataset, config)
                self.train_filename_pairs.extend(train_set.filename_pairs)
        for a_val_dataset in dict_file_group_spec[DataloaderKW.VALIDATION]:
            if a_val_dataset[DataloaderKW.TYPE] == "FILES":
                val_set = Files3DDataset(a_val_dataset, config)
                self.val_filename_pairs.extend(val_set.filename_pairs)
        for a_test_dataset in dict_file_group_spec[DataloaderKW.TEST]:
            if a_test_dataset[DataloaderKW.TYPE] == "FILES":
                test_set = Files3DDataset(a_test_dataset, config)
                self.test_filename_pairs.extend(test_set.filename_pairs)

    def validate_update_2d_train_val_test_dataset_and_filename_pairs(self, dict_file_group_spec):
        # Instantiate all the 2D Train/Val/Test Datasets

        # Given a JSON file, try to load the file pairing from it.
        assert dict_file_group_spec.get(DataloaderKW.TYPE).upper() == "FILES", \
            f"Invalid DataLoader type specified: {dict_file_group_spec.get(DataloaderKW.TYPE)}"

        # handle simple json specifications
        self.train_filename_pairs = self.parse_simple_spec_dict_for_filename_pairs(
            dict_file_group_spec.get(DataloaderKW.TRAINING)
        )

        self.val_filename_pairs = self.parse_simple_spec_dict_for_filename_pairs(
            dict_file_group_spec.get(DataloaderKW.VALIDATION)
        )

        self.test_filename_pairs = self.parse_simple_spec_dict_for_filename_pairs(
            dict_file_group_spec.get(DataloaderKW.TEST)
        )

        if not self.train_filename_pairs and not self.val_filename_pairs and not self.test_filename_pairs:
            raise ValueError("After parsing the loader dictionary, no valid file pairs were found in the dataset group.")

        # handle complex jSON specifications
        self.parse_complex_spec_dict_for_filename_pairs(
            dict_file_group_spec, DataloaderKW.TRAINING_TEST
        )

        self.parse_complex_spec_dict_for_filename_pairs(
            dict_file_group_spec, DataloaderKW.TRAINING_VALIDATION
        )

        self.parse_complex_spec_dict_for_filename_pairs(
            dict_file_group_spec, DataloaderKW.TRAINING_VALIDATION_TEST
        )

    def validate_complex_dataset(self, a_complex_dataset):
        """
        Parse a JSON dictionary of COMPLEX type: i.e. train_val, validate_test, train_val_test etc. and validate it.
        Args:
            a_complex_dataset:

        Returns:

        """
        pass


    def parse_complex_spec_dict_for_filename_pairs(self, a_complex_spec_dict: dict, complex_keyword: str):
        """
        Parse a JSON dictionary of SIMPLE type: i.e. train, validate, test etc. and validate it.
        Args:
            a_simple_spec_dict:

        Returns:

        """
        if complex_keyword not in a_complex_spec_dict:
            # Do nothing, as we will not be modifying the self.train_filename_pairs,
            # self.val_filename_pairs, self.test_filename_pairs
            return

        if complex_keyword == DataloaderKW.TRAINING_VALIDATION:
            # modifying the self.train_filename_pairs, self.val_filename_pairs
            pass

        elif complex_keyword == DataloaderKW.TRAINING_TEST:
            # modifying the self.train_filename_pairs, self.test_filename_pairs
            pass

        elif complex_keyword == DataloaderKW.TRAINING_VALIDATION_TEST:
            # modifying the self.train_filename_pairs, self.val_filename_pairs, self.test_filename_pairs
            pass

        else:
            raise ValueError("Unknown Complex Data Loader Type")

    def parse_simple_spec_dict_for_filename_pairs(self, a_simple_loader_json: dict) -> list:
        """Load the sub dictionary within the json file and return the dictionary"""

        filename_pairs = []

        if a_simple_loader_json is None:
            return filename_pairs

        if a_simple_loader_json.get(DataloaderKW.MISSING_FILES_HANDLE) == FileMissingHandleKW.SKIP:
            drop_missing = True
        else:
            drop_missing = False

        if DataloaderKW.IMAGE_GROUND_TRUTH in a_simple_loader_json.keys():
            filename_pairs = self.validate_simple_dataset(a_simple_loader_json, drop_missing)

        return filename_pairs

    def validate_simple_dataset(self, loader_json: dict, drop_missing: bool) -> list:
        """
        Parse a JSON dictionary of simple type: i.e. train, validate, test etc. and validate it.
        Args:
            a_simple_dataset:

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
            assert len(a_subject_image_ground_truth_pair) == 2, \
                f"Either subject images list or groundtruth files list are missing!"

            # Validate and trim Subject files list
            list_subject_specific_images: list = a_subject_image_ground_truth_pair[0]
            if len(list_subject_specific_images) > self.n_expected_input:
                logger.warning(f"HIGHER number of image files found than expected, for subject specification "
                               f"{a_subject_image_ground_truth_pair}. Expected {self.n_expected_input} "
                               f"but found {len(list_subject_specific_images)} from {list_subject_specific_images}")
                list_subject_specific_images = list_subject_specific_images[:self.n_expected_input]

            # Validate and trim Ground Truth files list
            list_subject_specific_gt: list = a_subject_image_ground_truth_pair[1]
            if len(list_subject_specific_gt) > self.n_expected_gt:
                logger.warning(f"HIGHER number of ground truth files found than expected, for subject specification "
                               f"{a_subject_image_ground_truth_pair}. Expected {self.n_expected_gt} "
                               f"but found {len(list_subject_specific_images)}. Using ground truth files {list_subject_specific_images[:self.n_expected_gt]}.")
                list_subject_specific_gt = list_subject_specific_gt[:self.n_expected_gt]

            # Go check every file and if any of them don't exist, skip the subject
            for subject_images_index, path_file in enumerate(
                    list_subject_specific_images
            ):
                list_subject_specific_images[subject_images_index] = ensure_absolute_path(
                    path_file, self.path_data
                )
                if not list_subject_specific_images[subject_images_index]:
                    skip_subject_flag = True

            # Exclude current subject
            if skip_subject_flag:
                continue

            if len(list_subject_specific_images) < self.n_expected_input:
                logger.warning(f"Fewer input files found than expected for subject specification "
                               f"{a_subject_image_ground_truth_pair}. Expected {self.n_expected_input} "
                               f"but found {len(list_subject_specific_images)} from {list_subject_specific_images}")
                if drop_missing:
                    continue

            if len(list_subject_specific_gt) < self.n_expected_gt:
                logger.warning(f"Fewer ground truth files found than expected, for subject specification "
                               f"{a_subject_image_ground_truth_pair}. Expected {self.n_expected_gt} "
                               f"but found {len(list_subject_specific_gt)} from {list_subject_specific_gt}")
                if drop_missing:
                    continue

            for gt_index, path_file in enumerate(list_subject_specific_gt):
                list_subject_specific_gt[gt_index] = ensure_absolute_path(
                    path_file, self.path_data
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

    def preview(self, verbose=False):
        """
        Preview the FINALIZED DataSetGroup included that has undergone validation and is ready to be used for training.
        Args:
            verbose: whether to print out the actual data path
        """
        logger.info(f"FileDatasetGroup: {self.name}")

        # Assume the simple types have been already assembled from complex types
        self.preview_simple_object_print("Train", self.train_filename_pairs, verbose)
        self.preview_simple_object_print("Val", self.val_filename_pairs, verbose)
        self.preview_simple_object_print("Test", self.test_filename_pairs, verbose)

    def preview_simple_object_print(self, file_set_type: str, file_name_pairs: list, verbose: bool):
        """
        Preview a simple data group object
        Args:
            file_set_type:
            file_name_pairs:
            verbose:
        """

        logger.info(f"\t{file_set_type} Files:")
        logger.info(
            f"\t\tFile {file_set_type} data set object has {len(file_name_pairs)} pairs of data files")
        if verbose:
            for pair_index, (image_files, gt_files, _, _) in enumerate(file_name_pairs):
                logger.info(f"\t\t\tImage Pair {pair_index}, Subject Image(s):")
                for a_image in image_files:
                    logger.info(f"\t\t\t\t{a_image}")
                logger.info(f"\t\t\tImage Pair {pair_index}, Ground Truth Image(s):")
                for a_gt in gt_files:
                    logger.info(f"\t\t\t\t{a_gt}")

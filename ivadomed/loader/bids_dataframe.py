import copy
import itertools
import os
import bids as pybids
import pandas as pd
from loguru import logger
from typing import List
from ivadomed.keywords import LoaderParamsKW, ROIParamsKW, ContrastParamsKW, BidsDataFrameKW
from pathlib import Path
import json


def write_derivatives_dataset_description(path_data: str):
    """Writes default dataset_description.json file if not found in path_data/derivatives folder

    Args:
        path_data (str): Path of the data

    Returns:

    """
    filename = "dataset_description"
    # Convert path_data to Path object for cross platform compatibility
    fname_data = Path(path_data)
    path_deriv_desc_file = fname_data / "derivatives" / f"{filename}.json"
    path_label_desc_file = fname_data / "derivatives" / "labels" / f"{filename}.json"

    # need to write default dataset_description.json file if not found
    if not path_deriv_desc_file.is_file() and not path_label_desc_file.is_file():
        with path_deriv_desc_file.open('w') as f:
            f.write(
                '{"Name": "Example dataset", "BIDSVersion": "1.0.2", "PipelineDescription": {"Name": "Example pipeline"}}')


class BidsDataframe:
    """
    This class aims to create a dataframe containing all BIDS image files in a list of path_data and their metadata.

    Args:
        loader_params (dict): Loader parameters, see :doc:`configuration_file` for more details.
        path_output (str): Output folder.
        derivatives (bool): If True, derivatives are indexed.

    Attributes:
        path_data (list): Paths to the BIDS datasets.
        bids_config (str): Path to the custom BIDS configuration file.
        target_suffix (list of str): List of suffix of targeted structures.
        roi_suffix (str): List of suffix of ROI masks.
        extensions (list of str): List of file extensions of interest.
        contrast_lst (list of str): List of the contrasts of interest.
        derivatives (bool): If True, derivatives are indexed.
        multichannel (bool): If True, we are deading with multi channel data.
        df (pd.DataFrame): Dataframe containing dataset information
    """

    def __init__(self, loader_params: dict, path_output: str, derivatives: bool):

        # Enable tracing of the input loader_params for easier debugging
        logger.trace(f"Consolidated Loader Parameters Received:\n {json.dumps(loader_params, indent=4)}")

        # paths_data from loader parameters
        self.paths_data: list = loader_params[LoaderParamsKW.PATH_DATA]

        # Get bids_config from loader parameters
        if LoaderParamsKW.BIDS_CONFIG not in loader_params:
            self.bids_config = None
        else:
            self.bids_config: str = loader_params[LoaderParamsKW.BIDS_CONFIG]

        # target_suffix and roi_suffix from loader parameters
        self.target_suffix: List[str] = copy.deepcopy(loader_params.get(LoaderParamsKW.TARGET_SUFFIX, []))

        # If `target_suffix` is a list of lists convert to list
        if any(isinstance(t, list) for t in self.target_suffix):
            self.target_suffix = list(itertools.chain.from_iterable(self.target_suffix))

        # This is different from target_suffix because suffix will eventually include ROI_suffix
        self.target_ground_truth: List[str] = copy.deepcopy(self.target_suffix)

        self.roi_suffix: str = loader_params[LoaderParamsKW.ROI_PARAMS][ROIParamsKW.SUFFIX]

        # If `roi_suffix` is not None, add to target_suffix
        if self.roi_suffix is not None:
            self.target_suffix.append(self.roi_suffix)

        # extensions from loader parameters
        if loader_params.get(LoaderParamsKW.EXTENSIONS):
            self.extensions: List[str] = loader_params[LoaderParamsKW.EXTENSIONS]
        else:
            self.extensions = [".nii", ".nii.gz"]

        # contrast_lst from loader parameters
        if not loader_params.get(LoaderParamsKW.CONTRAST_PARAMS):
            error_message = "Required target contrast parameters not found. Please CAREFULLY review JSON configuration"
            "file to ensure that at least one medical imaging modality/contrast is specified!"
            logger.error(error_message)
            raise ValueError(error_message)
        elif ContrastParamsKW.CONTRAST_LIST not in loader_params[LoaderParamsKW.CONTRAST_PARAMS]:
            self.contrast_lst: List[str] = []
        else:
            self.contrast_lst: List[str] = loader_params[LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.CONTRAST_LIST]

        # derivatives
        self.derivatives: bool = derivatives

        # Multichannel, if key not present, it is assumed to be false.
        self.multichannel: bool = bool(loader_params.get(LoaderParamsKW.MULTICHANNEL))

        # The source of ground truth, a string to look for in file name
        # Note that this may be a list of MORE than one element! E.g. "target_suffix": ["_seg-myelin-manual", "_seg-axon-manual"],
        if not self.target_suffix:
            raise ValueError(f'Loader parameters `{LoaderParamsKW.TARGET_SUFFIX}` is '
                             f"missing and the pattern string must be EXPLICITLY specified to identify the ground"
                             f"truth.")

        self.target_sessions: List[int] = loader_params.get(LoaderParamsKW.TARGET_SESSIONS)

        if not self.target_sessions:
            self.target_sessions = []
            logger.warning(f"Loader parameters {LoaderParamsKW.TARGET_SESSIONS} is missing. Presuming  single session "
                           f"study without session parameters. Multisession analyses needs to EXPLICITLY be specified: "
                           f"e.g. {LoaderParamsKW.TARGET_SESSIONS}: [1, 2]")
        else:
            # Convert them all to string
            self.target_sessions = list(map(str, self.target_sessions))

        # Create dataframe, which contain much information like file path, base file, extensions. etc
        self.df: pd.DataFrame = pd.DataFrame()
        self.create_bids_dataframe()

        # Save dataframe as csv file
        self.save(str(Path(path_output, "bids_dataframe.csv")))

    def create_bids_dataframe(self):
        """Generate the dataframe by first walk through the self.path_data to build the pybids.BIDSLayoutIndexer before
        converting that bids layout into a DATAFRAME which we add several custom columns
        """

        # Suppress a Future Warning from pybids about leading dot included in 'extension' from version 0.14.0
        # The config_bids.json file used matches the future behavior
        # TODO: when reaching version 0.14.0, remove the following line
        pybids.config.set_option('extension_initial_dot', True)

        # First pass over the data to create the self.df data
        # Filter for those files that has
        # 1) subject files of chosen contrasts and extensions,
        # 2) derivative files of chosen target_suffix from loader parameters
        self.df = self.first_inclusive_pass_data_frame_creation()

        if self.df.empty:
            # Raise error and exit if no subject files are found in any path data
            raise RuntimeError("No subject files found. Check selection of parameters in configuration JSON"
                               " and datasets compliance with BIDS specification. ")

        # Drop duplicated rows based on all columns except 'path'
        # Keep first occurrence
        columns = self.df.columns.to_list()
        columns.remove('path')
        self.df = self.df[~(self.df.astype(str).duplicated(subset=columns, keep='first'))]

        # If indexing of derivatives is true, do a second pass to filter for ONLY data that has self target_suffix ground truth labels
        if self.derivatives:
            self.df = self.second_exclusive_pass_dataframe_creation_removing_subjects_without_derivatives(self.df)

            # No derivatives, then no need to do target session either.
            # If multiple target sessions are specified, we filter for subjects that 1) doesn't have those sessions.
            if self.target_sessions:
                self.df = self.third_exclusive_pass_df_creation_check_modalities_sessions_combinations(self.df)

        # Reset index
        self.df.reset_index(drop=True, inplace=True)

        # Drop columns with all null values
        self.df.dropna(axis=1, inplace=True, how='all')

    def first_inclusive_pass_data_frame_creation(self) -> pd.DataFrame:
        """
        Conduct the first pass through the data and create a BIDS DataFrame.
        1) Across all self.paths_data
        2) Force index/microsopcy/CT scans
        3) Single (Non-Multisession) Version: must have one of the many CONTRASTS, one of the many EXTENSIONS,
        4) Multi-sessions Version: must have one of the many CONTRASTS, one of the many SESSIONS, one of the many EXTENSIONS

        This initial pass is MEANT to be inclusive. 2nd and 3rd passes are suppose to prune this data frame further.

        Returns: the first pass BIDS Dataframe.
        """

        # Avoid using side effect to update the self.df and force explicit assignment.
        first_pass_data_frame: pd.DataFrame = pd.DataFrame()

        # For path data objects that is included:
        for path_data in self.paths_data:
            path_data = Path(path_data, '')

            # Initialize BIDSLayoutIndexer and BIDSLayout
            # validate=True by default for both indexer and layout, BIDS-validator is not skipped
            list_force_index = self.construct_force_index_list(path_data, )
            indexer = pybids.BIDSLayoutIndexer(force_index=list_force_index)

            if self.derivatives:
                write_derivatives_dataset_description(str(path_data))

            layout = pybids.BIDSLayout(str(path_data), config=self.bids_config, indexer=indexer,
                                       derivatives=self.derivatives)

            # Transform layout to dataframe with all entities and json metadata
            # Stage 0
            # As per pyBIDS, derivatives don't include parsed entities, only the "path" column
            df_stage0: pd.DataFrame = layout.to_df(metadata=True)

            # Add filename column
            df_stage0.insert(1, BidsDataFrameKW.FILENAME, df_stage0[BidsDataFrameKW.PATH].apply(os.path.basename))

            # Stage 1: Drop rows with `json`, `tsv` and `LICENSE` files in case no extensions are provided in config file for filtering
            df_stage1 = df_stage0[
                ~df_stage0[BidsDataFrameKW.FILENAME].str.endswith(tuple(['.json', '.tsv', 'LICENSE']))]

            # Stage 2 Update dataframe with
            # 1) SUBJECTIVE files of chosen contrasts and extensions (SINGLE SESSION VERSION, no session filtering)
            if not self.target_sessions:
                df_filtered_subject_files_of_chosen_contrasts_and_extensions = (
                        ~df_stage1[BidsDataFrameKW.PATH].str.contains(
                            BidsDataFrameKW.DERIVATIVES)  # not derivative. Must be SUBJECTIVE data
                        & df_stage1[BidsDataFrameKW.SUFFIX].str.contains(
                    '|'.join(self.contrast_lst))  # must have one of the relevant contrast
                        & df_stage1[BidsDataFrameKW.EXTENSION].str.contains('|'.join(self.extensions))
                )
            # 1) SUBJECTIVE files of chosen contrasts and extensions (MULTI-SESSION VERSION, filter for data that are only
            # with the relevant sessions (i.e. cannot have missing session data)
            else:
                df_filtered_subject_files_of_chosen_contrasts_and_extensions = (
                        ~df_stage1[BidsDataFrameKW.PATH].str.contains(
                            BidsDataFrameKW.DERIVATIVES)  # not derivative. Must be SUBJECTIVE data
                        & df_stage1[BidsDataFrameKW.SUFFIX].str.contains(
                    '|'.join(self.contrast_lst))  # must have one of the relevant contrast
                        & df_stage1[BidsDataFrameKW.SESSION].str.contains(
                    '|'.join(self.target_sessions))  # must have one of the relevant targeted sessions
                        & df_stage1[BidsDataFrameKW.EXTENSION].str.contains('|'.join(self.extensions))
                )

            # and with 2) DERIVATIVE files of chosen target_suffix from loader parameters
            filter_derivative_files_of_chosen_target_suffix = (
                    df_stage1[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)  # must be derivatives
                    & df_stage1[BidsDataFrameKW.FILENAME].str.contains('|'.join(self.target_suffix))
                # don't care about session here as the ground truth can technically from ANY session
                # (assumed all session / contrast aligned)
            )

            # Stage 2 End Combine them together.
            df_stage2 = df_stage1[
                df_filtered_subject_files_of_chosen_contrasts_and_extensions
                | filter_derivative_files_of_chosen_target_suffix
                ]

            # WARNING if there are nothing other than derivative data (i.e. no subject files are found in path_data)
            if df_stage2[~df_stage2[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)].empty:
                logger.critical(f"No subject files were found in '{path_data}' dataset during FIRST PASS. "
                                f"Skipping dataset.")
                # first_pass_data_frame as an empty dataframe gets returned!

            else:
                # Add tsv files metadata to dataframe
                df_stage3 = self.add_tsv_metadata(df_stage2, str(path_data), layout)

                # TODO: check if other files are needed for EEG and DWI

                # Merge the default empty first_pass_data_frame with outer join to construct the proper output data frame
                first_pass_data_frame = pd.concat([first_pass_data_frame, df_stage3], join='outer', ignore_index=True)

        return first_pass_data_frame

    def construct_force_index_list(self, path_data: Path) -> list:
        """
        Examine the path given and clearly outline the files that needs to be forcibly indexed.
        Args:
            path_data (Path): the Path representation of where data is to conduct the screening to help build the list
            of files to be forcibly indexed

        Returns:

        """
        # TODO: remove force indexing of microscopy files after BEP microscopy is merged in BIDS
        # TODO: remove force indexing of CT-scan files after BEP CT-scan is merged in BIDS
        ext_microscopy = ('.png', '.tif', '.tiff', '.ome.tif', '.ome.tiff', '.ome.tf2', '.ome.tf8', '.ome.btf')
        ext_ct = ('.nii.gz', '.nii')
        suffix_ct = ('ct', 'CT')

        list_force_index: list = []

        for path_object in path_data.glob('**/*'):

            # Early exit if path object is not a file
            if not path_object.is_file():
                continue

            # Microscopy
            subject_path_index = len(path_data.parts)
            subject_path = path_object.parts[subject_path_index]
            file_name = path_object.name
            parent_folder_name = path_object.parent.name

            # Force index for samples tsv and json files, and for subject subfolders containing microscopy files
            # based on extensions.
            if file_name == "samples.tsv" or file_name == "samples.json":
                list_force_index.append(file_name)

            # If not subject data skip check against micropscy/CT-scan.
            if not subject_path.startswith('sub'):
                continue

            # Force index microscopy data
            if (file_name.endswith(ext_microscopy) and (parent_folder_name == "microscopy" or
                                                        parent_folder_name == "micr")):
                list_force_index.append(str(Path(*path_object.parent.parts[subject_path_index:])))

            # Force index of subject subfolders containing CT-scan files under "anat" or "ct" folder based on
            # extensions and modality suffix.
            if (file_name.endswith(ext_ct) and file_name.split('.')[0].endswith(suffix_ct) and
                    (parent_folder_name == "anat" or parent_folder_name == "ct")):
                list_force_index.append(str(Path(*path_object.parent.parts[subject_path_index:])))

        return list_force_index

    def second_exclusive_pass_dataframe_creation_removing_subjects_without_derivatives(self,
                                                                                       first_df: pd.DataFrame) -> pd.DataFrame:
        """
        Further filtering the dataframes based on those that has the appropriate target ground truth derivatives.
        Args:
            first_df (pd.DataFrame): DataFrame generated from FIRST pass inclusive search

        Returns:
            second_pass_dataframe (pd.DataFrame): DataFrame filtered for:
            1) subjects with at least one of each self.suffix

        """
        # Get:
        # list of subjects that has derivatives
        # list of ALL the available derivatives (all subjects, flat list)
        has_deriv, deriv = self.get_subjects_with_derivatives()

        # Default return empty data frames.
        second_pass_dataframe: pd.DataFrame = pd.DataFrame()

        # Filter dataframe to keep
        if has_deriv:
            second_pass_dataframe = first_df[
                # 1) subjects derivatives and
                first_df[BidsDataFrameKW.FILENAME].str.contains('|'.join(has_deriv))
                # 2) all known derivatives only
                | first_df[BidsDataFrameKW.FILENAME].str.contains('|'.join(deriv))
                ]
        else:
            # Raise error and exit if no derivatives are found for any subject files
            raise RuntimeError(
                "Not a single derivative was found at 2nd pass filtering for applicable subject specific derivatives. "
                "Training MUST at least have some ground truth labels.")

        return second_pass_dataframe

    def third_exclusive_pass_df_creation_check_modalities_sessions_combinations(self, df_next):
        """
        Go through the BIDSDataFrame, exclude subject missing sessions and mandatory modalities.
        Args:
            df_next:
        """

        # Exclude subject with missing sessions or missing modalities
        list_excluded_subjects: list = []
        list_unique_subjects: list = df_next[BidsDataFrameKW.SUBJECT].dropna().unique().tolist()

        for subject in list_unique_subjects:

            # Select all files with that subject
            df_subject: pd.DataFrame = df_next[df_next[BidsDataFrameKW.SUBJECT] == subject]

            # Select all subjects' unique sessionS
            current_subject_sessions: list = df_subject[BidsDataFrameKW.SESSION].unique().tolist()

            # Target Session Set:
            set_required_sessions: set = set(list(map(int, self.target_sessions)))
            set_current_subject_sessions: set = set(list(map(int, current_subject_sessions)))

            # When missing session set, exclude the subjects.
            if not set_required_sessions.issubset(set_current_subject_sessions):
                list_excluded_subjects.append(subject)
                continue

            # Otherwise, go through each of TARGET SESSIONS for this subject.
            for session in self.target_sessions:
                # Retrieve all subject's information with matching session
                df_session = df_subject[df_subject[BidsDataFrameKW.SESSION] == session]
                # Validate modality information.
                if self.validate_modality(df_session):
                    pass
                else:
                    # Exclude subject
                    list_excluded_subjects.append(subject)
                    # Go to next subject.
                    break

        # Commit the exclusion of subjects.
        df_next = self.exclude_subjects(list_excluded_subjects, df_next)

        return df_next

    def validate_modality(self, data_frame: pd.DataFrame):
        """
        Using the given dataframe to perform a validation on if all modality data are present.
        Args:
            data_frame: filtered dataframes to be checked for contrast compliance.
        """
        # Retrieve the unique set of suffix/modalities for this SUBJECT + SESSION
        set_contrasts_available: set = set(data_frame[BidsDataFrameKW.SUFFIX].unique().tolist())

        # Find the set from the requirement
        set_required_contrast_list: set = set(self.contrast_lst)

        # When missing a contrast, exclude the subject.
        if not set_required_contrast_list.issubset(set_contrasts_available):
            return False
        else:
            return True

    def exclude_subjects(self, list_exclude_subject: list, df_next: pd.DataFrame):
        """
        Given a BIDSDataFrame, exclude all source files and derivative files that contain the subject IDs from the list
        of the excluded subjects.
        Args:
            list_exclude_subject:
            df_next:

        Returns:

        """

        # When given an empty list of invalid list of subjects for exclusion, early return.
        if not list_exclude_subject:
            return df_next

        filter_subject_exclude_subject_list = (
            # Non-derivative files
                ~df_next[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)
                # Subject contains any of the excluded subjects
                & ~df_next[BidsDataFrameKW.SUBJECT].str.contains('|'.join(list_exclude_subject), na=False)
        )
        filter_derivative_exclude_subject_list = (
            # Derivative files
                df_next[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)
                # Subject contains any of the excluded subjects
                & ~df_next[BidsDataFrameKW.FILENAME].str.contains('|'.join(list_exclude_subject))
        )
        # Use "OR" to combined them together.
        df_next = df_next[filter_derivative_exclude_subject_list
                          | filter_subject_exclude_subject_list]
        return df_next

    def add_tsv_metadata(self, df: pd.DataFrame, path_data: str, layout: pybids.BIDSLayout):

        """Add tsv files metadata to dataframe.
        Args:
            layout (BIDSLayout): pybids BIDSLayout of the indexed files of the path_data
        """

        # Add participant_id column, and metadata from participants.tsv file if present
        # Uses pybids function
        df['participant_id'] = "sub-" + df['subject']
        if layout.get_collections(level='dataset'):
            df_participants = layout.get_collections(level='dataset', merge=True).to_df()
            df_participants.drop(['suffix'], axis=1, inplace=True)
            df = pd.merge(df, df_participants, on='subject', suffixes=("_x", None), how='left')

        # Add sample_id column if sample column exists, and add metadata from samples.tsv file if present
        # TODO: use pybids function after BEP microscopy is merged in BIDS
        if 'sample' in df:
            df['sample_id'] = "sample-" + df['sample']
        fname_samples = Path(path_data, "samples.tsv")
        if fname_samples.exists():
            df_samples = pd.read_csv(str(fname_samples), sep='\t')
            df = pd.merge(df, df_samples, on=['participant_id', 'sample_id'], suffixes=("_x", None),
                          how='left')

        # Add metadata from all _sessions.tsv files, if present
        # Uses pybids function
        if layout.get_collections(level='subject'):
            df_sessions = layout.get_collections(level='subject', merge=True).to_df()
            df_sessions.drop(['suffix'], axis=1, inplace=True)
            df = pd.merge(df, df_sessions, on=['subject', 'session'], suffixes=("_x", None), how='left')

        # Add metadata from all _scans.tsv files, if present
        # TODO: use pybids function after BEP microscopy is merged in BIDS
        # TODO: verify merge behavior with EEG and DWI scans files, tested with anat and microscopy only
        df_scans = pd.DataFrame()
        for path_object in Path(path_data).glob("**/*"):
            if path_object.is_file():
                if path_object.name.endswith("scans.tsv"):
                    df_temp = pd.read_csv(str(path_object), sep='\t')
                    df_scans = pd.concat([df_scans, df_temp], ignore_index=True)
        if not df_scans.empty:
            df_scans['filename'] = df_scans['filename'].apply(os.path.basename)
            df = pd.merge(df, df_scans, on=['filename'], suffixes=("_x", None), how='left')

        return df

    def get_subjects_with_derivatives(self) -> (list, list):
        """Get lists of subject filenames with available derivatives.

        Returns:
            list, list: subject filenames having derivatives, available derivatives filenames.
        """

        # Update self.df to remove not qualified subjects
        self.prune_subjects_without_ground_truth()

        # All subject's file name with the MATCHING modalities
        subject_filenames: list = self.get_subject_fnames()  # all known subjects' file names
        deriv_filenames = self.get_deriv_fnames()  # all known derivatives across all subjects

        subjects_with_derivatives = []  # subject filenames having derivatives
        all_available_derivatives = []  # the available derivatives filenames from ALL subjects... not co-indexed with
        # subjects_with_derivatives???

        for subject_filename in subject_filenames:
            # For the current subject, get its list of subject specific available derivatives
            # These COULD be across sessions/modalities!
            list_subject_available_derivatives: list = self.get_derivatives(subject_filename, deriv_filenames)

            # Early stop if no derivatives found. Go to next subject.
            if not list_subject_available_derivatives:
                logger.warning(f"Missing ALL derivatives for {subject_filename}. Skipping.")
                continue
            # Check for all the mandatory modalities and ensure that they

            at_least_one_ground_truth_found: bool = self.check_for_at_least_one_one_ground_truth(
                list_subject_available_derivatives, subject_filename
            )
            if not at_least_one_ground_truth_found:
                logger.warning(f"Missing at mandatory derivatives for {subject_filename}. Skipping.")
                continue

            # Handle subject specific roi_suffix related filtering?
            if self.roi_suffix is not None:
                all_available_derivatives, subjects_with_derivatives = self.include_first_roi_specific_derivative(
                    all_available_derivatives,
                    subjects_with_derivatives,
                    list_subject_available_derivatives,
                    subject_filename)
            else:
                all_available_derivatives, subjects_with_derivatives = self.include_first_subject_specific_derivatives(
                    all_available_derivatives,
                    subjects_with_derivatives,
                    list_subject_available_derivatives,
                    subject_filename)

        return subjects_with_derivatives, all_available_derivatives

    def prune_subjects_without_ground_truth(self):
        """Remove any subject from self.df that has no ground truth information.
        """
        list_exclude_subject = []
        # all known derivatives across all subjects
        deriv_filenames: list = self.get_deriv_fnames()
        # Get list of unique subjects so can get derivatives associated with each one.
        list_unique_subjects: list = self.df[BidsDataFrameKW.SUBJECT].dropna().unique().tolist()
        # Update list of excluded subjects by going through each unique subject and ensure at least ONE ground truth
        # exists
        for subject in list_unique_subjects:

            # Get the list of derivatives matching the subject.
            list_subject_available_derivatives = list(filter(
                lambda a_file_name:  # the file name from the deriv_filenames to be checked for `subject in`
                subject in a_file_name,  # boolean for the filtering function
                deriv_filenames  # the list to iterate through.
            ))

            at_least_one_ground_truth_found: bool = self.check_for_at_least_one_one_ground_truth(
                list_subject_available_derivatives, subject
            )
            if not at_least_one_ground_truth_found:
                list_exclude_subject.append(subject)

        # Exclude the subject's data.
        self.df = self.exclude_subjects(list_exclude_subject, self.df)

    def check_for_at_least_one_one_ground_truth(self, list_subject_available_derivatives: list, subject_filename: str):
        """
        Go through each derivative, check to ensure it is in at least ONE file of the list_subject_available_derivatives

        Args:
            list_subject_available_derivatives (list):
            subject_filename (str):

        Returns:

        """

        for target_derivative in self.target_ground_truth:
            target_derivative_found: bool = any(
                list(
                    # Map the check function to the file list
                    map(
                        lambda a_derivative: target_derivative in a_derivative,
                        list_subject_available_derivatives
                    )
                )
            )
            if target_derivative_found:
                return True

        logger.warning(f"Not even one one ground truth found for {subject_filename}.")
        return False

    def include_first_subject_specific_derivatives(self, all_available_derivatives: list,
                                                   subjects_with_derivatives: list,
                                                   list_subject_available_derivatives: list, subject_filename: str):
        """ Include the first derivative/ground truth by including it in the all_available_derivatives and subjects_with_derivatives list.
        Args:
            all_available_derivatives:
            subjects_with_derivatives:
            list_subject_available_derivatives:
            subject_filename:
        """
        # Go through each suffix.
        target_suffix: str

        target_ground_truth_suffix_found = False
        # Here uses target ground truth, which EXCLUDES ROI suffix.
        for target_suffix in self.target_ground_truth:

            # Generate the list of related derivatives.
            list_target_suffix_derivatives: list = list(
                # Map the check function to the file list
                filter(
                    lambda a_derivative: target_suffix in a_derivative,
                    list_subject_available_derivatives
                )
            )

            # If the list is emtpy, warn but check next
            if not list_target_suffix_derivatives:
                logger.warning(f"No target derivatives found for {target_suffix} for {subject_filename} during "
                               f"inclusion check. Skipping.")
                continue
            else:
                # Update derivative list
                all_available_derivatives.extend(list_target_suffix_derivatives)
                # Mark for inclusion.
                target_ground_truth_suffix_found = True

        # Update this list only ONCE, if any issue, arises, it would have returned earlier (e.g. empty list)
        if target_ground_truth_suffix_found:
            subjects_with_derivatives.append(subject_filename)

        return all_available_derivatives, subjects_with_derivatives

    def include_first_roi_specific_derivative(self, all_available_derivatives: list, subjects_with_derivatives: list,
                                              list_subject_available_derivatives: list, subject_filename: str):
        """ Include the first derivative/ground truth by including it in the all_available_derivatives and subjects_with_derivatives list.
        Args:
            all_available_derivatives:
            subjects_with_derivatives:
            list_subject_available_derivatives:
            subject_filename:
        """

        # If we did not find
        if self.roi_suffix not in ('|'.join(list_subject_available_derivatives)):
            logger.warning(f"Missing roi_suffix {self.roi_suffix} for {subject_filename}. Skipping.")
            return all_available_derivatives, subjects_with_derivatives

        # Go through each suffix.
        target_suffix: str

        target_ground_truth_suffix_found = False
        # Here uses target_suffix, which INCLUDES ROI suffix.
        for target_suffix in self.target_suffix:

            # Generate the list of related derivatives.
            list_target_suffix_derivatives: list = list(
                # Map the check function to the file list
                filter(
                    lambda a_derivative: target_suffix in a_derivative,
                    list_subject_available_derivatives
                )
            )

            # If the list is emtpy, warn but check next
            if not list_target_suffix_derivatives:
                logger.warning(f"No target derivatives found for {target_suffix} for {subject_filename} during "
                               f"inclusion check. Skipping.")
                continue
            else:
                # Update derivative list
                all_available_derivatives.extend(list_target_suffix_derivatives)
                # Mark for inclusion.
                target_ground_truth_suffix_found = True

        # Update this list only ONCE, if any issue, arises, it would have returned earlier (e.g. empty list)
        if target_ground_truth_suffix_found:
            subjects_with_derivatives.append(subject_filename)

        return all_available_derivatives, subjects_with_derivatives

    def get_subject_fnames(self) -> list:
        """Get the list of BIDS validated subject filenames in dataframe.

        Returns:
            list: subject filenames.
        """
        return self.df[~self.df['path'].str.contains('derivatives')]['filename'].to_list()

    def get_all_subject_ids_with_derivatives(self) -> list:
        """Get the list of subject filenames in dataframe regardless of modalities.

        Returns:
            list: subject IDs
        """
        subject_field_from_every_derivative_files: list = self.df[~self.df['path'].str.contains('derivatives')][
            'subject'].to_list()
        unique_subjects_with_derivatives: list = list(set(subject_field_from_every_derivative_files))
        return unique_subjects_with_derivatives

    def get_deriv_fnames(self) -> list:
        """Get the list of BIDS validated derivative filenames in dataframe.

        Returns:
            list: derivative filenames.
        """
        return self.df[self.df['path'].str.contains('derivatives')]['filename'].tolist()

    def get_derivatives(self, subject_filename: str, deriv_filenames: List[str]) -> List[str]:
        """Given a subject fname full path information, return list of AVAILABLE derivative filenames for the subject
        Args:
            subject_filename (str): Subject filename, NOT a full path. e.g. sub-ms01_ses-01_FLAIR
            deriv_filenames (list of str): list of ALL BIDS validated files from the derivative folder

        Returns:
            list: a list of all the derivative filenames for that particular subject
            There shouldn't be too many derivatives per subject.
        """

        # e.g. sub-ms01
        subject_id = subject_filename.split('_')[0]

        # Obtain the list of files that are directly matching the subject fname.
        # Could be empty.
        list_derived_matching_subject_fname = list(filter(
            lambda a_file_name:
            # as long as 1) the subject_id
            # we consider that a valid ground truth/derivatives
            subject_id in a_file_name,
            # apply to the entire list of deriv_filenames
            deriv_filenames
        ))

        # Sort in place for alphabetical order output.
        list_derived_matching_subject_fname.sort()

        # If multichannel is not used, just do the simple filtering for that particular subject.
        if not self.multichannel:
            return list_derived_matching_subject_fname

        # If no derivative found, return empty list.
        if not list_derived_matching_subject_fname:
            logger.warning(f"No derivatives found for {subject_filename}")
            return list_derived_matching_subject_fname

        return list_derived_matching_subject_fname

    def save(self, path: str):
        """Save the dataframe into a csv file.
        Args:
            path (str): Path to csv file.
        """
        try:
            self.df.to_csv(path, index=False)
            logger.info("Dataframe has been saved in {}.".format(path))
        except FileNotFoundError:
            logger.error("Wrong path, bids_dataframe.csv could not be saved in {}.".format(path))

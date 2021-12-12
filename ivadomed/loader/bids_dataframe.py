import copy
import itertools
import os
import bids as pybids
import pandas as pd
from loguru import logger
from typing import List
from ivadomed.keywords import LoaderParamsKW, ROIParamsKW, ContrastParamsKW, BidsDataFrameKW
from pathlib import Path
from pprint import pformat


def write_derivatives_dataset_description(path_data: str):
    """Writes default dataset_description.json file if not found in path_data/derivatives folder

    Args:
        path_data (str): Path of the data

    Returns:

    """
    filename = "dataset_description"
    # Convert path_data to Path object for cross platform compatibility
    fname_data = Path(path_data)
    deriv_desc_file = str(fname_data / "derivatives" / f"{filename}.json")
    label_desc_file = str(fname_data / "derivatives" / "labels" / f"{filename}.json")

    # need to write default dataset_description.json file if not found
    if not os.path.isfile(deriv_desc_file) and not os.path.isfile(label_desc_file):
        f = open(deriv_desc_file, 'w')
        f.write(
            '{"Name": "Example dataset", "BIDSVersion": "1.0.2", "PipelineDescription": {"Name": "Example pipeline"}}')
        f.close()


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
        target_suffix (list of str): List of suffix of targetted structures.
        roi_suffix (str): List of suffix of ROI masks.
        extensions (list of str): List of file extensions of interest.
        contrast_lst (list of str): List of the contrasts of interest.
        derivatives (bool): If True, derivatives are indexed.
        multichannel (bool): If True, we are deading with multi channel data.
        df (pd.DataFrame): Dataframe containing dataset information
    """

    def __init__(self, loader_params: dict, path_output: str, derivatives: bool):

        # paths_data from loader parameters
        self.paths_data: list = loader_params[LoaderParamsKW.PATH_DATA]

        # Get bids_config from loader parameters
        if LoaderParamsKW.BIDS_CONFIG not in loader_params:
            self.bids_config = None
        else:
            self.bids_config: str = loader_params[LoaderParamsKW.BIDS_CONFIG]

        # target_suffix and roi_suffix from loader parameters
        self.target_suffix: List[str] = copy.deepcopy(loader_params[LoaderParamsKW.TARGET_SUFFIX])

        # If `target_suffix` is a list of lists convert to list
        if any(isinstance(t, list) for t in self.target_suffix):
            self.target_suffix = list(itertools.chain.from_iterable(self.target_suffix))

        self.roi_suffix: str = loader_params[LoaderParamsKW.ROI_PARAMS][ROIParamsKW.SUFFIX]

        # If `roi_suffix` is not None, add to target_suffix
        if self.roi_suffix is not None:
            self.target_suffix.append(self.roi_suffix)

        # extensions from loader parameters
        if loader_params[LoaderParamsKW.EXTENSIONS]:
            self.extensions: List[str] = loader_params[LoaderParamsKW.EXTENSIONS]
        else:
            self.extensions = [".nii", ".nii.gz"]

        # contrast_lst from loader parameters
        if ContrastParamsKW.CONTRAST_LST not in loader_params[LoaderParamsKW.CONTRAST_PARAMS]:
            self.contrast_lst: List[str] = []
        else:
            self.contrast_lst: List[str] = loader_params[LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.CONTRAST_LST]


        # derivatives
        self.derivatives: bool = derivatives

        # Multichannel, if key not present, it is assumed to be false.
        self.multichannel: bool = bool(loader_params.get(LoaderParamsKW.MULTICHANNEL))

        # The single source of ground truth, a string to look for in file name
        self.target_ground_truth: str = loader_params.get(LoaderParamsKW.TARGET_GROUND_TRUTH)
        if not self.target_ground_truth:
            raise ValueError(f'Loader parameters "{LoaderParamsKW.TARGET_GROUND_TRUTH}" is '
                             f"missing and the pattern string must be EXPLICITLY specified to identify the ground"
                             f"truth.")

        self.target_sessions: List[int] = loader_params.get(LoaderParamsKW.TARGET_SESSIONS)

        if not self.target_sessions:
            self.target_sessions = []
            logger.warning(f"Loader parameters {LoaderParamsKW.TARGET_SESSIONS} is missing. Presuming  single session "
                           f"study without session parameters. Multisession analyses needs to be specified: e.g. "
                           f"{LoaderParamsKW.TARGET_SESSIONS}: [1, 2]")
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

        # First pass over the data to create the self.df
        # Filter for those files that has
        # 1) subject files of chosen contrasts and extensions,
        # 2) derivative files of chosen target_suffix from loader parameters
        self.df = self.first_inclusive_pass_data_frame_creation()

        if self.df.empty:
            # Raise error and exit if no subject files are found in any path data
            raise RuntimeError("No subject files found. Check selection of parameters in config.json"
                               " and datasets compliance with BIDS specification.")

        # Drop duplicated rows based on all columns except 'path'
        # Keep first occurrence
        columns = self.df.columns.to_list()
        columns.remove('path')
        self.df = self.df[~(self.df.astype(str).duplicated(subset=columns, keep='first'))]

        # If indexing of derivatives is true, do a second pass to filter for ONLY data that has self target_ground truth
        if self.derivatives:
            self.df = self.second_exclusive_pass_dataframe_creation_removing_subjects_without_derivatives(self.df)

            # If multiple target sessions are specified, we filter for subjects that 1) doesn't have those sessions.
            if self.target_sessions:
                self.df = self.third_exclusive_pass_df_creation_check_modalities_sessions_combinations(self.df)

        # Reset index
        self.df.reset_index(drop=True, inplace=True)

        # Drop columns with all null values
        self.df.dropna(axis=1, inplace=True, how='all')

    def second_exclusive_pass_dataframe_creation_removing_subjects_without_derivatives(self, first_df: pd.DataFrame) -> pd.DataFrame:
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

        second_pass_dataframe: pd.DataFrame = pd.DataFrame()

        # Filter dataframe to keep
        if has_deriv:
            second_pass_dataframe = first_df[
                # 1) subjects files and
                first_df[BidsDataFrameKW.FILENAME].str.contains('|'.join(has_deriv))
                # 2) all known derivatives only
                | first_df[BidsDataFrameKW.FILENAME].str.contains('|'.join(deriv))
            ]
        else:
            # Raise error and exit if no derivatives are found for any subject files
            raise RuntimeError("Not a single derivative was found. Training MUST at least have some ground truth labels.")

        return second_pass_dataframe

    def first_inclusive_pass_data_frame_creation(self) -> pd.DataFrame:
        """
        Conduct the first pass through the data and create a BIDS DataFrame.
        1) Across all self.paths_data
        2) Force index/microsopcy/CT scans
        3) Non-Multisession Version: must have one of the many CONTRASTS, one of the many EXTENSIONS,
        4) Multisession Version: must have one of the many CONTRASTS, one of the many SESSIONS, one of the many EXTENSIONS

        Returns: the first past BIDS Dataframe.
        """

        # Avoid using side effect to update the self.df and force explicit assignment.
        first_pass_data_frame: pd.DataFrame = pd.DataFrame()

        # For path data objects that is included:
        for path_data in self.paths_data:
            path_data = Path(path_data, '')

            # Initialize BIDSLayoutIndexer and BIDSLayout
            # validate=True by default for both indexer and layout, BIDS-validator is not skipped
            # Force index for samples tsv and json files, and for subject subfolders containing microscopy files based on extensions.
            # Force index of subject subfolders containing CT-scan files under "anat" or "ct" folder based on extensions and modality suffix.
            # TODO: remove force indexing of microscopy files after BEP microscopy is merged in BIDS
            # TODO: remove force indexing of CT-scan files after BEP CT-scan is merged in BIDS
            ext_microscopy = ('.png', '.tif', '.tiff', '.ome.tif', '.ome.tiff', '.ome.tf2', '.ome.tf8', '.ome.btf')
            ext_ct = ('.nii.gz', '.nii')
            suffix_ct = ('ct', 'CT')
            force_index = []
            for path_object in path_data.glob('**/*'):
                if path_object.is_file():
                    # Microscopy
                    subject_path_index = len(path_data.parts)
                    subject_path = path_object.parts[subject_path_index]
                    if path_object.name == "samples.tsv" or path_object.name == "samples.json":
                        force_index.append(path_object.name)
                    if (path_object.name.endswith(ext_microscopy) and (path_object.parent.name == "microscopy" or
                            path_object.parent.name == "micr") and subject_path.startswith('sub')):
                        force_index.append(str(Path(*path_object.parent.parts[subject_path_index:])))
                    # CT-scan
                    if (path_object.name.endswith(ext_ct) and path_object.name.split('.')[0].endswith(suffix_ct) and
                            (path_object.parent.name == "anat" or path_object.parent.name == "ct") and
                            subject_path.startswith('sub')):
                        force_index.append(str(Path(*path_object.parent.parts[subject_path_index:])))
            indexer = pybids.BIDSLayoutIndexer(force_index=force_index)

            if self.derivatives:
                write_derivatives_dataset_description(str(path_data))

            layout = pybids.BIDSLayout(str(path_data), config=self.bids_config, indexer=indexer,
                                       derivatives=self.derivatives)

            # Transform layout to dataframe with all entities and json metadata
            # As per pybids, derivatives don't include parsed entities, only the "path" column
            df_next = layout.to_df(metadata=True)

            # Add filename column
            df_next.insert(1, BidsDataFrameKW.FILENAME, df_next[BidsDataFrameKW.PATH].apply(os.path.basename))

            # Drop rows with json, tsv and LICENSE files in case no extensions are provided in config file for filtering
            df_next = df_next[~df_next[BidsDataFrameKW.FILENAME].str.endswith(tuple(['.json', '.tsv', 'LICENSE']))]

            # Update dataframe with
            # 1) subject files of chosen contrasts and extensions (single session version, no session filtering)
            if not self.target_sessions:
                filter_subject_files_of_chosen_contrasts_and_extensions = (
                        ~df_next[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)
                        & df_next[BidsDataFrameKW.SUFFIX].str.contains('|'.join(self.contrast_lst))
                        & df_next[BidsDataFrameKW.EXTENSION].str.contains('|'.join(self.extensions))
                )
            # 1) subject files of chosen contrasts and extensions (multi-session version, filter for data that are only
            # within the relevant session)
            else:
                filter_subject_files_of_chosen_contrasts_and_extensions = (
                        ~df_next[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)
                        & df_next[BidsDataFrameKW.SUFFIX].str.contains('|'.join(self.contrast_lst))
                        & df_next[BidsDataFrameKW.SESSION].str.contains('|'.join(self.target_sessions))
                        & df_next[BidsDataFrameKW.EXTENSION].str.contains('|'.join(self.extensions))
                )

            # and with 2) derivative files of chosen target_suffix from loader parameters
            filter_derivative_files_of_chosen_target_suffix = (
                df_next[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)
                & df_next[BidsDataFrameKW.FILENAME].str.contains('|'.join(self.target_suffix))
            )

            # Combine them together.
            df_next: pd.DataFrame = df_next[
                filter_subject_files_of_chosen_contrasts_and_extensions
                | filter_derivative_files_of_chosen_target_suffix
            ]

            # Warning if no subject files are found in path_data
            if df_next[~df_next[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)].empty:
                logger.warning(f"No subject files were found in '{path_data}' dataset during FIRST PASS. "
                               f"Skipping dataset.")

            else:
                # Add tsv files metadata to dataframe
                df_next = self.add_tsv_metadata(df_next, path_data, layout)

                # TODO: check if other files are needed for EEG and DWI

                # Merge dataframes with outer join: i.e. avoid duplicates.
                first_pass_data_frame = pd.concat([first_pass_data_frame, df_next], join='outer', ignore_index=True)

        return first_pass_data_frame

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

            # Select all subjects' unique sessions
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

                # Retrieve the unique set of suffix/modalities for this SUBJECT + SESSION
                set_current_subject_suffixes: set = set(df_session[BidsDataFrameKW.SUFFIX].unique().tolist())
                # Find the set from the requirement
                set_current_contrast_list: set = set(self.contrast_lst)

                # When missing a contrast, exclude the subject.
                if not set_current_contrast_list.issubset(set_current_subject_suffixes):
                    list_excluded_subjects.append(subject)
                    break

        df_next = self.exclude_subjects(list_excluded_subjects, df_next)

        return df_next

    def exclude_subjects(self, list_exclude_subject: list, df_next: pd.DataFrame):
        """
        Given a BIDSDataFrame, exclude all source files and derivative files that contain the subject IDs from the list
        of the excluded subjects.
        Args:
            list_exclude_subject:
            df_next:

        Returns:

        """

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
        # OR combined them to gether.
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
        list_exclude_subject = []
        deriv_filenames: list = self.get_deriv_fnames()  # all known derivatives across all subjects
        list_unique_subjects: list = self.df[BidsDataFrameKW.SUBJECT].dropna().unique().tolist()

        for subject in list_unique_subjects:
            list_derived_matching_subject = list(filter(
                lambda a_file_name:
                subject in a_file_name,
                deriv_filenames
            ))

            list_derived_target_suffix = list(map(
                lambda filename: '_' + filename.split('_')[-1].split('.')[0], list_derived_matching_subject))
            # derivatives for each subject must contain at least ONE of EACH of the target_suffix
            if not set(self.target_suffix).issubset(set(list_derived_target_suffix)):
                list_exclude_subject.append(subject)

        self.df = self.exclude_subjects(list_exclude_subject, self.df)

        subject_filenames: list = self.get_subject_fnames()  # all known subjects' file names
        deriv_filenames = self.get_deriv_fnames()  # all known derivatives across all subjects

        has_deriv = []  # subject filenames having derivatives
        deriv = []  # the available derivatives filenames from ALL subjects... not co-indexed with has_deriv???

        for subject_filename in subject_filenames:
            # For the current subject, get its list of subject specific available derivatives
            # These COULD be across sessions/modalities!
            list_subject_available_derivatives: list = self.get_derivatives(subject_filename, deriv_filenames)

            # Early stop if no derivatives found. Go to next subject.
            if not list_subject_available_derivatives:
                logger.warning(f"Missing ALL derivatives for {subject_filename}. Skipping.")
                continue
            # Check for all the mandatory modalities and ensure that they

            at_least_one_of_each_derivatives_found: bool = self.check_for_at_least_one_of_each_derivative_suffix(
                list_subject_available_derivatives, subject_filename
            )
            if not at_least_one_of_each_derivatives_found:
                logger.warning(f"Missing at least some mandatory derivatives for {subject_filename}. Skipping.")
                continue

            # Handle subject specific roi_suffix related filtering?
            if self.roi_suffix is not None:
                self.include_first_roi_specific_derivative(deriv, has_deriv, list_subject_available_derivatives,
                                                           subject_filename)
            else:
                self.include_first_subject_specific_derivatives(deriv, has_deriv, list_subject_available_derivatives,
                                                                subject_filename)

        return has_deriv, deriv

    def check_for_at_least_one_of_each_derivative_suffix(self, list_subject_available_derivatives: list, subject_filename: str):
        """
        Go through each derivative, check to ensure it is in at least ONE file of the list_subject_available_derivatives

        Args:
            list_subject_available_derivatives (list):
            subject_filename (str):

        Returns:

        """

        for target_derivative in self.target_suffix:
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
                logger.warning(f"No target derivatives found for {target_derivative} for {subject_filename}. Skipping.")
                return False

        return True

    def include_first_subject_specific_derivatives(self, deriv: list, has_deriv: list,
                                                   list_subject_available_derivatives: list, subject_filename: str):
        """ Include the first derivative/ground truth by including it in the deriv and has_deriv list.
        Args:
            deriv:
            has_deriv:
            list_subject_available_derivatives:
            subject_filename:
        """
        # Go through each suffix.
        target_suffix: str
        for target_suffix in self.target_suffix:

            # Generate the list of related derivatives.
            list_target_suffix_derivatives: list = list(
                    # Map the check function to the file list
                    filter(
                        lambda a_derivative: target_suffix in a_derivative,
                        list_subject_available_derivatives
                    )
                )

            # If the list is emtpy, warn .
            if not list_target_suffix_derivatives:
                logger.critical(f"No target derivatives found for {target_suffix} for {subject_filename} during "
                                f"inclusion check. Skipping.")
                return

            # If the list hast multiple ground truth
            if len(list_target_suffix_derivatives) > 1:
                    logger.warning(f"When evaluating {subject_filename},"
                                   f"more than one ground truth matching '{target_suffix}' found. "
                                   f"Expected ONE ground truth PER subject PER target_suffix ACROSS sessions/modalities:"
                                   f"\n{pformat(list_subject_available_derivatives)}")
                    logger.critical(f"First one is chosen: {list_subject_available_derivatives[0]}")

            # Update derivative list
            deriv.append(list_subject_available_derivatives[0])

        # Update this list only ONCE, if any issue, arises, it would have returned earlier (e.g. empty list)
        has_deriv.append(subject_filename)

    def include_first_roi_specific_derivative(self, deriv: list, has_deriv: list,
                                              list_subject_available_derivatives: list, subject_filename: str):
        """ Include the first derivative/ground truth by including it in the deriv and has_deriv list.
        Args:
            deriv:
            has_deriv:
            list_subject_available_derivatives:
            subject_filename:
        """

        # If one of the ground truth has ROI_suffix in it, we prefer that one.
        if self.roi_suffix not in ('|'.join(list_subject_available_derivatives)):
            logger.warning(f"Missing roi_suffix {self.roi_suffix} for {subject_filename}. Skipping.")
            return

        # This additional layer of filtering for only ROI suffix.
        list_roi_derivatives = list(filter(
            lambda a_derivative_filename: self.roi_suffix in a_derivative_filename,
            list_subject_available_derivatives
        ))

        # Go through each suffix.
        target_suffix: str
        for target_suffix in self.target_suffix:

            # Generate the list of related derivatives.
            list_target_suffix_derivatives: list = list(
                # Map the check function to the file list
                filter(
                    lambda a_derivative: target_suffix in a_derivative,
                    list_roi_derivatives
                )
            )

            # If the list is emtpy, warn .
            if not list_target_suffix_derivatives:
                logger.critical(f"No target derivatives matching {self.roi_suffix} found for {target_suffix} for "
                                f"{subject_filename} during inclusion check. Skipping.")
                return

            # If the list hast multiple ground truth
            if len(list_target_suffix_derivatives) > 1:
                logger.warning(f"When evaluating {subject_filename},"
                               f"more than one derivative matching '{target_suffix}' found. "
                               f"Expected ONE ground truth PER subject PER target_suffix ACROSS sessions/modalities:"
                               f"\n{pformat(list_subject_available_derivatives)}")
                logger.critical(f"First one is chosen: {list_subject_available_derivatives[0]}")

            # Update derivative list
            deriv.append(list_subject_available_derivatives[0])

        # Update this list only ONCE, if any issue, arises, it would have returned earlier (e.g. empty list)
        has_deriv.append(subject_filename)


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

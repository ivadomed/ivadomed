import copy
import itertools
import os

import bids as pybids
import numpy as np
import pandas as pd
from loguru import logger
from typing import List
from ivadomed.keywords import LoaderParamsKW, ROIParamsKW, ContrastParamsKW, BidsDataFrameKW
from pathlib import Path


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
        f.write('{"Name": "Example dataset", "BIDSVersion": "1.0.2", "PipelineDescription": {"Name": "Example pipeline"}}')
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
        self.extensions: List[str] = loader_params[LoaderParamsKW.EXTENSIONS]

        # contrast_lst from loader parameters
        if ContrastParamsKW.CONTRAST_LIST not in loader_params[LoaderParamsKW.CONTRAST_PARAMS]:
            self.contrast_lst: List[str] = []
        else:
            self.contrast_lst: List[str] = loader_params[LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.CONTRAST_LIST]

        # derivatives
        self.derivatives: bool = derivatives

        # Multichannel, if key not present, it is assumed to be false.
        self.multichannel: bool = bool(loader_params.get(LoaderParamsKW.MULTICHANNEL))

        # Create dataframe, which contain much information like file path, base file, extensions. etc
        self.df: pd.DataFrame = pd.DataFrame()
        self.create_bids_dataframe()

        # Save dataframe as csv file
        self.save(os.path.join(path_output, "bids_dataframe.csv"))

    def create_bids_dataframe(self):
        """Generate the dataframe by first walk through the self.path_data to build the pybids.BIDSLayoutIndexer before
        converting that bids layout into a DATAFRAME which we add several custom columns
        """

        # Suppress a Future Warning from pybids about leading dot included in 'extension' from version 0.14.0
        # The config_bids.json file used matches the future behavior
        # TODO: when reaching version 0.14.0, remove the following line
        pybids.config.set_option('extension_initial_dot', True)

        for path_data in self.paths_data:
            path_data = os.path.join(path_data, '')

            # Initialize BIDSLayoutIndexer and BIDSLayout
            # validate=True by default for both indexer and layout, BIDS-validator is not skipped
            # Force index for samples tsv and json files, and for subject subfolders containing microscopy files based on extensions.
            # Force index of subject subfolders containing CT-scan files under "anat" or "ct" folder based on extensions and modality suffix.
            # TODO: remove force indexing of microscopy files after BEP microscopy is merged in BIDS
            # TODO: remove force indexing of CT-scan files after BEP CT-scan is merged in BIDS
            ext_microscopy = ('.png', '.ome.tif', '.ome.tiff', '.ome.tf2', '.ome.tf8', '.ome.btf')
            ext_ct = ('.nii.gz', '.nii')
            suffix_ct = ('ct', 'CT')
            force_index = []
            for root, dirs, files in os.walk(path_data):
                for file in files:
                    # Microscopy
                    if file == "samples.tsv" or file == "samples.json":
                        force_index.append(file)
                    if (file.endswith(ext_microscopy) and os.path.basename(root) == "microscopy" and
                            (root.replace(path_data, '').startswith("sub"))):
                        force_index.append(os.path.join(root.replace(path_data, '')))
                    # CT-scan
                    if (file.endswith(ext_ct) and file.split('.')[0].endswith(suffix_ct) and
                            (os.path.basename(root) == "anat" or os.path.basename(root) == "ct") and
                            (root.replace(path_data, '').startswith("sub"))):
                        force_index.append(os.path.join(root.replace(path_data, '')))
            indexer = pybids.BIDSLayoutIndexer(force_index=force_index)

            if self.derivatives:
                write_derivatives_dataset_description(path_data)

            layout = pybids.BIDSLayout(path_data, config=self.bids_config, indexer=indexer,
                                       derivatives=self.derivatives)

            # Transform layout to dataframe with all entities and json metadata
            # As per pybids, derivatives don't include parsed entities, only the "path" column
            df_next = layout.to_df(metadata=True)

            # Add filename column
            df_next.insert(1, BidsDataFrameKW.FILENAME, df_next[BidsDataFrameKW.PATH].apply(os.path.basename))

            # Drop rows with json, tsv and LICENSE files in case no extensions are provided in config file for filtering
            df_next = df_next[~df_next[BidsDataFrameKW.FILENAME].str.endswith(tuple(['.json', '.tsv', 'LICENSE']))]

            # Update dataframe with 1) subject files of chosen contrasts and extensions,
            # and with 2) derivative files of chosen target_suffix from loader parameters
            filter_subject_files_of_chosen_contrasts_and_extensions = (
                    ~df_next[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)
                    & df_next[BidsDataFrameKW.SUFFIX].str.contains('|'.join(self.contrast_lst))
                    & df_next[BidsDataFrameKW.EXTENSION].str.contains('|'.join(self.extensions))
            )
            filter_derivative_files_of_chosen_target_suffix = (
                    df_next[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)
                    & df_next[BidsDataFrameKW.FILENAME].str.contains('|'.join(self.target_suffix))
            )
            df_next = df_next[filter_subject_files_of_chosen_contrasts_and_extensions
                              | filter_derivative_files_of_chosen_target_suffix]

            if df_next[~df_next[BidsDataFrameKW.PATH].str.contains(BidsDataFrameKW.DERIVATIVES)].empty:
                # Warning if no subject files are found in path_data
                logger.warning(f"No subject files were found in '{path_data}' dataset. Skipping dataset.")

            else:
                # Add tsv files metadata to dataframe
                df_next = self.add_tsv_metadata(df_next, path_data, layout)

                # TODO: check if other files are needed for EEG and DWI

                # Merge dataframes with outer join: i.e. avoid duplicates.
                self.df = pd.concat([self.df, df_next], join='outer', ignore_index=True)

        if self.df.empty:
            # Raise error and exit if no subject files are found in any path data
            raise RuntimeError("No subject files found. Check selection of parameters in config.json"
                               " and datasets compliance with BIDS specification.")

        # Drop duplicated rows based on all columns except 'path'
        # Keep first occurrence
        columns = self.df.columns.to_list()
        columns.remove('path')
        self.df = self.df[~(self.df.astype(str).duplicated(subset=columns, keep='first'))]

        # If indexing of derivatives is true
        if self.derivatives:

            # Get
            # list of subjects that has derivatives
            # list of ALL the available derivatives (all subjects, flat list)
            has_deriv, deriv = self.get_subjects_with_derivatives()

            # Filter dataframe to keep 1) subjects files and 2) all known derivatives only
            if has_deriv:
                self.df = self.df[self.df[BidsDataFrameKW.FILENAME].str.contains('|'.join(has_deriv))
                                  | self.df[BidsDataFrameKW.FILENAME].str.contains('|'.join(deriv))]
            else:
                # Raise error and exit if no derivatives are found for any subject files
                raise RuntimeError("Derivatives not found.")

        # Reset index
        self.df.reset_index(drop=True, inplace=True)

        # Drop columns with all null values
        self.df.dropna(axis=1, inplace=True, how='all')

    def add_tsv_metadata(self, df, path_data, layout):

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
        fname_samples = os.path.join(path_data, "samples.tsv")
        if os.path.exists(fname_samples):
            df_samples = pd.read_csv(fname_samples, sep='\t')
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
        for root, dirs, files in os.walk(path_data):
            for file in files:
                if file.endswith("scans.tsv"):
                    df_temp = pd.read_csv(os.path.join(root, file), sep='\t')
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
        subject_fnames: list = self.get_subject_fnames()  # all known subjects' fnames
        deriv_fnames: list = self.get_deriv_fnames()  # all known derivatives
        has_deriv = []  # subject filenames having derivatives
        deriv = []  # the available derivatives filenames from ALL subjects... not coindexed with has_deriv???

        for subject_fname in subject_fnames:
            # For the current subject, get its list of subject specific available derivatives
            list_subject_available_derivatives: list = self.get_derivatives(subject_fname, deriv_fnames)

            # Early stop if no derivatives found. Go to next subject.
            if not list_subject_available_derivatives:
                logger.warning(f"Missing derivatives for {subject_fname}. Skipping.")
                continue

            # Handle subject specific roi_suffix related filtering?
            if self.roi_suffix is not None:
                if self.roi_suffix in ('|'.join(list_subject_available_derivatives)):
                    has_deriv.append(subject_fname)
                    deriv.extend(list_subject_available_derivatives)
                else:
                    logger.warning(f"Missing roi_suffix {self.roi_suffix} for {subject_fname}. Skipping.")
            else:
                has_deriv.append(subject_fname)
                deriv.extend(list_subject_available_derivatives)

            # Warn about when there is a missing derivative
            for target in self.target_suffix:
                if target not in str(list_subject_available_derivatives) and target != self.roi_suffix:
                    logger.warning(f"Missing target_suffix {target} for {subject_fname}")

        return has_deriv, deriv

    def get_subject_fnames(self) -> list:
        """Get the list of subject filenames in dataframe.

        Returns:
            list: subject filenames.
        """
        return self.df[~self.df['path'].str.contains('derivatives')]['filename'].to_list()

    def get_deriv_fnames(self) -> list:
        """Get the list of derivative filenames in dataframe.

        Returns:
            list: derivative filenames.
        """
        return self.df[self.df['path'].str.contains('derivatives')]['filename'].tolist()

    def get_derivatives(self, subject_fname: str, deriv_fnames: List[str]) -> List[str]:
        """Given a subject fname full path information, return list of AVAILABLE derivative filenames for the subject
        Args:
            subject_fname (str): Subject filename, NOT a full path.
            deriv_fnames (list of str): List of derivative filenames.

        Returns:
            list: a list of all the derivative filenames for that particular subject
        """
        prefix_fname = subject_fname.split('.')[0]
        # Obtain the list of files that are directly matching the subject fname.
        # Could be empty.
        list_derived_matching_subject_fname = list(filter(lambda a_file_name: prefix_fname in a_file_name, deriv_fnames))

        # If multichannel is not used, just do the simple filtering for that particular subject.
        if not self.multichannel:
            return list_derived_matching_subject_fname

        # If no derivative found, return empty list.
        if not list_derived_matching_subject_fname:
            return list_derived_matching_subject_fname

        # First check - look for ground truth of other contrasts
        for contrast in self.contrast_lst:
            prefix_fname = subject_fname.split('_')[0] + "_" + contrast
            list_derived_matching_subject_fname = list(filter(lambda a_file_name: prefix_fname in a_file_name, deriv_fnames))
            if list_derived_matching_subject_fname:
                break

        # Return if found.
        if list_derived_matching_subject_fname:
            return list_derived_matching_subject_fname

        # Early return if no session information found in file name.
        if "_ses-" not in subject_fname:
            return list_derived_matching_subject_fname

        # This point onward, the file is GUARNTEED to have "_ses-" in the file name.
        # Get Subject name?
        subject = subject_fname.split('_')[0]

        # Get Session ID
        sess_id = subject_fname.split('_')[1]

        # Get the last bit BEFORE the extension?
        # e.g. sub-unf01_T1w_lesion-manual.nii.gz
        # would return lesion-manual
        # e.g. sub-unf01_T1w_ses-01.nii.gz
        # would return ses-01
        # sub-X_ses-1_acq-Y_T1w.nii.gz
        # sub-X_ses-1_acq-Z_T1w.nii.gz
        contrast_id = subject_fname.split('_')[-1].split(".")[0]

        all_subject_deriv = [d for d in deriv_fnames if subject in d]

        # Second check - in case of ses-* file structure, check for the derivatives in the sessions folder
        session_list: np.ndarray = np.unique([d.split("_")[1] for d in all_subject_deriv])

        # Session list is reordered to first check for the same contrast-type in the other sessions
        if len(session_list) > 1:
            re_ordered_session_lst: list = [sess_id] + session_list.remove(sess_id)
        else:
            re_ordered_session_lst: list = session_list

        # For Each session in the re-ordered session list.
        for session in re_ordered_session_lst:
            # Contrast list is reordered to first check for the same contrast-type in the other sessions
            if len(self.contrast_lst) > 1:
                re_ordered_contrast_lst = [contrast_id] + self.contrast_lst.remove(contrast_id)
            else:
                re_ordered_contrast_lst = self.contrast_lst
            for contrast in re_ordered_contrast_lst:
                new_prefix_fname = subject_fname.split('.')[0].split('_')[0] + \
                                   "_" + session + "_" + contrast
                list_derived_matching_subject_fname = [d for d in deriv_fnames if new_prefix_fname in d]
                if list_derived_matching_subject_fname:
                    logger.info(
                        "Multichannel training: No derivative found for {} "
                        "- Assigning derivatives from {}".format(
                            subject_fname,
                            np.unique(["_".join([x.split("_")[0], x.split("_")[1], contrast])
                                       for x in list_derived_matching_subject_fname])[0])
                    )
                    break

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
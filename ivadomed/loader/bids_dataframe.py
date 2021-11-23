import copy
import itertools
import os

import bids as pybids
import pandas as pd
from loguru import logger
from pathlib import Path


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
        df (pd.DataFrame): Dataframe containing dataset information
    """

    def __init__(self, loader_params, path_output, derivatives):

        # paths_data from loader parameters
        self.paths_data = loader_params['path_data']

        # bids_config from loader parameters
        self.bids_config = None if 'bids_config' not in loader_params else loader_params['bids_config']

        # target_suffix and roi_suffix from loader parameters
        self.target_suffix = copy.deepcopy(loader_params['target_suffix'])

        # If `target_suffix` is a list of lists convert to list
        if any(isinstance(t, list) for t in self.target_suffix):
            self.target_suffix = list(itertools.chain.from_iterable(self.target_suffix))

        self.roi_suffix = loader_params['roi_params']['suffix']

        # If `roi_suffix` is not None, add to target_suffix
        if self.roi_suffix is not None:
            self.target_suffix.append(self.roi_suffix)

        # extensions from loader parameters
        self.extensions = loader_params['extensions'] if loader_params['extensions'] else [".nii", ".nii.gz"]

        # contrast_lst from loader parameters
        self.contrast_lst = [] if 'contrast_lst' not in loader_params['contrast_params'] \
            else loader_params['contrast_params']['contrast_lst']

        # derivatives
        self.derivatives = derivatives

        # Create dataframe
        self.df = pd.DataFrame()
        self.create_bids_dataframe()

        # Save dataframe as csv file
        self.save(str(Path(path_output, "bids_dataframe.csv")))

    def create_bids_dataframe(self):
        """Generate the dataframe."""

        # Suppress a Future Warning from pybids about leading dot included in 'extension' from version 0.14.0
        # The config_bids.json file used matches the future behavior
        # TODO: when reaching version 0.14.0, remove the following line
        pybids.config.set_option('extension_initial_dot', True)

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
                self.write_derivatives_dataset_description(path_data)

            layout = pybids.BIDSLayout(str(path_data), config=self.bids_config, indexer=indexer,
                                       derivatives=self.derivatives)

            # Transform layout to dataframe with all entities and json metadata
            # As per pybids, derivatives don't include parsed entities, only the "path" column
            df_next = layout.to_df(metadata=True)

            # Add filename column
            df_next.insert(1, 'filename', df_next['path'].apply(os.path.basename))

            # Drop rows with json, tsv and LICENSE files in case no extensions are provided in config file for filtering
            df_next = df_next[~df_next['filename'].str.endswith(tuple(['.json', '.tsv', 'LICENSE']))]

            # Update dataframe with subject files of chosen contrasts
            # and with derivative files of chosen target_suffix
            df_next = df_next[(~df_next['path'].str.contains('derivatives')
                               & df_next['suffix'].str.contains('|'.join(self.contrast_lst)))
                              | (df_next['path'].str.contains('derivatives')
                                 & df_next['filename'].str.contains('|'.join(self.target_suffix)))]

            # Update dataframe with files of chosen extensions
            df_next = df_next[df_next['filename'].str.endswith(tuple(self.extensions))]

            # Warning if no subject files are found in path_data
            if df_next[~df_next['path'].str.contains('derivatives')].empty:
                logger.warning("No subject files were found in '{}' dataset. Skipping dataset.".format(path_data))
            else:
                # Add tsv files metadata to dataframe
                df_next = self.add_tsv_metadata(df_next, path_data, layout)

                # TODO: check if other files are needed for EEG and DWI

                # Merge dataframes
                self.df = pd.concat([self.df, df_next], join='outer', ignore_index=True)

        if self.df.empty:
            # Raise error and exit if no subject files are found in any path data
            raise RuntimeError("No subject files found. Check selection of parameters in config.json"
                               " and datasets compliance with BIDS specification.")

        # Drop duplicated rows based on all columns except 'path'
        # Keep first occurence
        columns = self.df.columns.to_list()
        columns.remove('path')
        self.df = self.df[~(self.df.astype(str).duplicated(subset=columns, keep='first'))]

        # If indexing of derivatives is true
        if self.derivatives:

            # Get list of subject files with available derivatives
            has_deriv, deriv = self.get_subjects_with_derivatives()

            # Filter dataframe to keep subjects files with available derivatives only
            if has_deriv:
                self.df = self.df[self.df['filename'].str.contains('|'.join(has_deriv))
                                  | self.df['filename'].str.contains('|'.join(deriv))]
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

    def get_subjects_with_derivatives(self):
        """Get lists of subject filenames with available derivatives.

        Returns:
            list, list: subject filenames having derivatives, available derivatives filenames.
        """
        subject_fnames = self.get_subject_fnames()
        deriv_fnames = self.get_deriv_fnames()
        has_deriv = []
        deriv = []

        for subject_fname in subject_fnames:
            available = self.get_derivatives(subject_fname, deriv_fnames)
            if available:
                if self.roi_suffix is not None:
                    if self.roi_suffix in ('|'.join(available)):
                        has_deriv.append(subject_fname)
                        deriv.extend(available)
                    else:
                        logger.warning("Missing roi_suffix {} for {}. Skipping."
                                       .format(self.roi_suffix, subject_fname))
                else:
                    has_deriv.append(subject_fname)
                    deriv.extend(available)
                for target in self.target_suffix:
                    if target not in str(available) and target != self.roi_suffix:
                        logger.warning("Missing target_suffix {} for {}".format(target, subject_fname))
            else:
                logger.warning("Missing derivatives for {}. Skipping.".format(subject_fname))

        return has_deriv, deriv

    def get_subject_fnames(self):
        """Get the list of subject filenames in dataframe.

        Returns:
            list: subject filenames.
        """
        return self.df[~self.df['path'].str.contains('derivatives')]['filename'].to_list()

    def get_deriv_fnames(self):
        """Get the list of derivative filenames in dataframe.

        Returns:
            list: derivative filenames.
        """
        return self.df[self.df['path'].str.contains('derivatives')]['filename'].tolist()

    def get_derivatives(self, subject_fname, deriv_fnames):
        """Return list of available derivative filenames for a subject filename.
        Args:
            subject_fname (str): Subject filename.
            deriv_fnames (list of str): List of derivative filenames.

        Returns:
            list: derivative filenames
        """
        prefix_fname = subject_fname.split('.')[0]
        return [d for d in deriv_fnames if prefix_fname in d]

    def save(self, path):
        """Save the dataframe into a csv file.
        Args:
            path (str): Path to csv file.
        """
        try:
            self.df.to_csv(path, index=False)
            logger.info("Dataframe has been saved in {}.".format(path))
        except FileNotFoundError:
            logger.error("Wrong path, bids_dataframe.csv could not be saved in {}.".format(path))

    def write_derivatives_dataset_description(self, path_data):
        """Writes default dataset_description.json file if not found in path_data/derivatives folder
        """
        path_data = Path(path_data).absolute()

        filename = 'dataset_description'
        path_deriv_desc_file = Path(f'{path_data}/derivatives/{filename}.json')
        path_label_desc_file = Path(f'{path_data}/derivatives/labels/{filename}.json')
        # need to write default dataset_description.json file if not found
        if not path_deriv_desc_file.is_file() and not path_label_desc_file.is_file():

            logger.warning(f"{path_deriv_desc_file} not found. Please ensure a full path is specified in the "
                           f"configuration file. Will attempt to create a place holder description file for now at"
                           f"{path_deriv_desc_file}.")
            with path_deriv_desc_file.open(mode='w') as f:
                f.write(
                    '{"Name": "Example dataset", '
                    '"BIDSVersion": "1.0.2", '
                    '"PipelineDescription": {"Name": "Example pipeline"}}'
                )

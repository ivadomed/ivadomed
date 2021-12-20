from typing import List

import pandas as pd
from tqdm import tqdm
import numpy as np
from ivadomed.loader import film as imed_film
from ivadomed.loader.mri2d_segmentation_dataset import MRI2DSegmentationDataset
from ivadomed.object_detection import utils as imed_obj_detect
from ivadomed.keywords import ROIParamsKW, ContrastParamsKW, ModelParamsKW, MetadataKW, SubjectDictKW, BidsDataFrameKW
from loguru import logger

class BidsDataset(MRI2DSegmentationDataset):
    """ BIDS specific dataset loader.

    Args:
        bids_df (BidsDataframe): Object containing dataframe with all BIDS image files and their metadata.
        subject_file_lst (list): Subject filenames list.
        target_suffix (list): List of suffixes for target masks.
        contrast_params (dict): Contains image contrasts related parameters.
        model_params (dict): Dictionary containing model parameters.
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        cache (bool): If the data should be cached in memory or not.
        transform (list): Transformation list (length 2) composed of preprocessing transforms (Compose) and transforms
            to apply during training (Compose).
        metadata_choice (str): Choice between "mri_params", "contrasts", the name of a column from the
            participants.tsv file, None or False, related to FiLM.
        slice_filter_fn (SliceFilter): Class that filters slices according to their content.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.
        multichannel (bool): If True, the input contrasts are combined as input channels for the model. Otherwise, each
            contrast is processed individually (ie different sample / tensor).
        object_detection_params (dict): Object dection parameters.
        task (str): Choice between segmentation or classification. If classification: GT is discrete values, \
            If segmentation: GT is binary mask.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
        truths are thresholded (0.5) after the data augmentation operations.
        is_input_dropout (bool): Return input with missing modalities.

    Attributes:
        filename_pairs (list): A list of tuples in the format (input filename list containing all modalities,ground \
            truth filename, ROI filename, metadata).
        metadata (dict): Dictionary containing FiLM metadata.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
            truths are thresholded (0.5) after the data augmentation operations.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.

    """

    def __init__(self, bids_df, subject_file_lst, target_suffix, contrast_params, model_params, slice_axis=2,
                 cache=True, transform=None, metadata_choice=False, slice_filter_fn=None, roi_params=None,
                 multichannel=False, object_detection_params=None, task="segmentation", soft_gt=False,
                 is_input_dropout=False):

        self.roi_params = roi_params if roi_params is not None else \
            {ROIParamsKW.SUFFIX: None, ROIParamsKW.SLICE_FILTER_ROI: None}
        self.soft_gt = soft_gt
        self.filename_pairs = []
        if metadata_choice == MetadataKW.MRI_PARAMS:
            self.metadata = {"FlipAngle": [], "RepetitionTime": [],
                             "EchoTime": [], "Manufacturer": []}

        # Sort subject_file_lst and create a sub-dataframe from bids_df containing only subjects from subject_file_lst
        subject_file_lst = sorted(subject_file_lst)
        df_subjects = bids_df.df[bids_df.df['filename'].isin(subject_file_lst)]

        # Create a dictionary with the number of subjects for each contrast of contrast_balance
        tot = {contrast: df_subjects['suffix'].str.fullmatch(contrast).value_counts()[True]
               for contrast in contrast_params[ContrastParamsKW.BALANCE].keys()}

        # Create a counter that helps to balance the contrasts
        contrast_counter = {contrast: 0 for contrast in contrast_params[ContrastParamsKW.BALANCE].keys()}

        # Get a list of subject_ids for multichannel_subjects (prefix filename without modality suffix and extension)
        subject_ids = []
        for subject in subject_file_lst:
            subject_ids.append(subject.split('.')[0].split('_')[0])
        subject_ids = sorted(list(set(subject_ids)))

        # Create multichannel_subjects dictionary for each subject_id
        multichannel_subjects = {}
        idx_dict = {}
        sess_dict = {}
        if multichannel:
            num_contrast = len(contrast_params[ContrastParamsKW.CONTRAST_LIST])
            session_list = np.unique([d.split("_")[1] for d in df_subjects['filename'] if "ses-" in d])

            for idx, contrast in enumerate(contrast_params[ContrastParamsKW.CONTRAST_LIST]):
                idx_dict[contrast] = idx

            for idx, session in enumerate(session_list):
                sess_dict[session] = idx
            if session_list.size != 0:
                multichannel_subjects = {subject: {"absolute_paths": [None] * num_contrast * len(session_list),
                                                   "deriv_path": None,
                                                   "roi_filename": None,
                                                   "metadata": [None] * num_contrast * len(session_list)}
                                         for subject in subject_ids}
            else:
                multichannel_subjects = {subject: {"absolute_paths": [None] * num_contrast,
                                                   "deriv_path": None,
                                                   "roi_filename": None,
                                                   "metadata": [None] * num_contrast}
                                         for subject in subject_ids}

        # Get all subjects path from bids_df for bounding box
        get_all_subj_path = bids_df.df[bids_df.df['filename']
                                .str.contains('|'.join(bids_df.get_subject_fnames()))]['path'].to_list()

        # Load bounding box from list of path
        bounding_box_dict = imed_obj_detect.load_bounding_boxes(object_detection_params,
                                                                get_all_subj_path,
                                                                slice_axis,
                                                                contrast_params[ContrastParamsKW.CONTRAST_LIST])

        # Get all derivatives filenames from bids_df
        all_deriv = bids_df.get_deriv_fnames()

        # Create filename_pairs
        for subject in tqdm(subject_file_lst, desc="Loading dataset"):
            df_sub, roi_filename, target_filename, metadata = self.create_filename_pair(multichannel_subjects, subject,
                                                                                        contrast_counter, tot, multichannel, df_subjects,
                                                                                        contrast_params, target_suffix,
                                                                                        all_deriv, bids_df, bounding_box_dict,
                                                                                        idx_dict, metadata_choice)
            # Reject check to skip current subject.
            if df_sub is None or target_filename is None or metadata is None:
                continue

            # Fill multichannel dictionary
            # subj_id is the filename without modality suffix and extension
            if multichannel:
                multichannel_subjects = self.fill_multichannel_dict(multichannel_subjects, subject, idx_dict, sess_dict,
                                                                    df_sub, roi_filename, target_filename, metadata)
            else:
                self.filename_pairs.append(([df_sub['path'].values[0]],
                                            target_filename, roi_filename, [metadata]))

        if multichannel:
            for subject in multichannel_subjects.values():
                if None not in subject["absolute_paths"]:
                    self.filename_pairs.append((subject["absolute_paths"], subject["deriv_path"],
                                                subject["roi_filename"], subject[SubjectDictKW.METADATA]))

        if not self.filename_pairs:
            raise Exception('No subjects were selected - check selection of parameters on config.json (e.g. center '
                            'selected + target_suffix)')

        length = model_params[ModelParamsKW.LENGTH_2D] if ModelParamsKW.LENGTH_2D in model_params else []
        stride = model_params[ModelParamsKW.STRIDE_2D] if ModelParamsKW.STRIDE_2D in model_params else []

        super().__init__(self.filename_pairs, length, stride, slice_axis, cache, transform, slice_filter_fn, task, self.roi_params,
                         self.soft_gt, is_input_dropout)

    def validate_derivative_path_to_update_target_filename(self,
                                                           derivative_path: str,
                                                           target_suffix: list or List[list],
                                                           target_filename: list,
                                                           ):
        """
        FOR the given derivative path, update target_filename array IF there is a match between the TARGET SUFFIX
        Args:
            target_suffix: list of target suffix to check.
            target_filename:
            derivative_path: string indicative of the path of a single derivative file

        Returns:

        """
        # Go through each suffix.
        for index, suffixes in enumerate(target_suffix):
            # If suffixes is a string, then only one rater annotation per class is available.
            if isinstance(suffixes, str):
                if suffixes in derivative_path:
                    target_filename[index] = derivative_path
            # Otherwise, multiple raters segmented the same class and we need to check EACH of them.
            elif isinstance(suffixes, list):
                for suffix in suffixes:
                    # Check if the suffix string is a part of the derivative_path string.
                    if suffix in derivative_path:
                        target_filename[index].append(derivative_path)


    def get_most_relevant_target_filename(self, subject_file_name: str, target_suffix: str or list, bids_df_derivatives: pd.DataFrame):
        """
        Among all potential ground truth out there across sessions, choose the most appropriate one based on either
        session match OR first sorted session.
        Args:
            subject_file_name: str, name of the subject modality file
            target_suffix: list of target suffix to check.
            bids_df_derivatives: dataframe of derivatives which are ground truth across SESSIONS and MODALITY.

        Returns:

        """
        # Empty instantiate target_filename and roi_filename to their respective type, simple or nested lists.
        if isinstance(target_suffix[0], str):
            target_filename = [None] * len(target_suffix)
        else:
            target_filename = [
                [] for _ in range(len(target_suffix))
            ]

        # If there is session in the original file data:
        if "_ses-" in subject_file_name:

            # String process to get the session information from subject.
            subject_session: str = ""
            name_parts = subject_file_name.split("_")
            for part in name_parts:
                if "ses-" in part:
                    subject_session = part
                    break

            # Further filter the bids_df_derivatives for matching sessions:
            list_session_matched_derivative_path: list = bids_df_derivatives[
                bids_df_derivatives[BidsDataFrameKW.FILENAME].str.contains(subject_session)
            ][BidsDataFrameKW.PATH].to_list()

            # Early empty return.
            if not list_session_matched_derivative_path:
                return target_filename

            list_derivative_path = list_session_matched_derivative_path

        # if not sort path (as session is already ascending) and use the EARLIEST session data.
        else:

            # Further filter the bids_df_derivatives for matching sessions:
            list_session_non_matched_derivative_path: list = bids_df_derivatives[BidsDataFrameKW.PATH].to_list()

            # Early empty return.
            if not list_session_non_matched_derivative_path:
                return target_filename

            list_derivative_path = list_session_non_matched_derivative_path

        # In the end, go through the respective derivative path, identify best target_file name
        for derivative_path in list_derivative_path:
            self.validate_derivative_path_to_update_target_filename(derivative_path, target_suffix, target_filename)


        return target_filename, list_derivative_path

    def create_metadata_dict(self, metadata, metadata_choice, df_sub, bids_df):
        # add custom data to metadata
        if metadata_choice not in df_sub.columns:
            raise ValueError("The following metadata cannot be found: {}. "
                                "Invalid metadata choice.".format(metadata_choice))
        metadata[metadata_choice] = df_sub[metadata_choice].values[0]
        # Create metadata dict for OHE
        data_lst = sorted(set(bids_df.df[metadata_choice].dropna().values))
        metadata_dict = {}
        for idx, data in enumerate(data_lst):
            metadata_dict[data] = idx
        metadata[MetadataKW.METADATA_DICT] = metadata_dict

    def fill_multichannel_dict(self, multichannel_subjects, subject, idx_dict, sess_dict, df_sub,
                               roi_filename, target_filename, metadata):

        if "ses-" not in subject:
            idx = idx_dict[df_sub['suffix'].values[0]]
            file_session = []
        else:
            file_session = subject.split("_")[1]
            idx = (len(sess_dict)-1)*sess_dict[file_session] + idx_dict[df_sub['suffix'].values[0]]

        subj_id = subject.split('.')[0].split('_')[0]
        multichannel_subjects[subj_id]["absolute_paths"][idx] = df_sub['path'].values[0]
        multichannel_subjects[subj_id]["deriv_path"] = target_filename
        multichannel_subjects[subj_id][SubjectDictKW.METADATA][idx] = metadata
        if roi_filename:
            multichannel_subjects[subj_id]["roi_filename"] = roi_filename
        return multichannel_subjects


    def create_filename_pair(self, multichannel_subjects, subject: str, contrast_counter, tot, multichannel, df_subjects, contrast_params,
                             target_suffix, all_deriv, bids_df, bounding_box_dict, idx_dict, metadata_choice):
        """

        Args:
            multichannel_subjects:
            subject: str, a file name, representing the subject data, has extension
            contrast_counter:
            tot:
            multichannel:
            df_subjects:
            contrast_params:
            target_suffix:
            all_deriv: all the derivatives for all subjects.
            bids_df:
            bounding_box_dict:
            idx_dict:
            metadata_choice:

        Returns:

        """

        # Filter to Get Subject specific dataframes
        df_sub = df_subjects.loc[df_subjects[BidsDataFrameKW.FILENAME] == subject]

        # ???
        # Training & Validation: do not consider the contrasts over the threshold contained in contrast_balance
        contrast = df_sub[BidsDataFrameKW.SUFFIX].values[0]
        if contrast in (contrast_params[ContrastParamsKW.BALANCE].keys()):
            contrast_counter[contrast] = contrast_counter[contrast] + 1
            if contrast_counter[contrast] / tot[contrast] > contrast_params[ContrastParamsKW.BALANCE][contrast]:
                return

        # Empty instantiate target_filename and roi_filename to their respective type, simple or nested lists.
        if isinstance(target_suffix[0], str):
            target_filename = [None] * len(target_suffix)
        else:
            target_filename = [
                [] for _ in range(len(target_suffix))
            ]
        roi_filename = None

        # Filter the dataframe, for specific derivative file names which matches ONE of the subject, among ALL derivatives
        # Note that this dataframe has NO OTHER INFORMATION, as they are FORCE INDEXED.
        bids_df_derivatives = bids_df.df[
            bids_df.df[BidsDataFrameKW.FILENAME].str.contains(
                '|'.join(bids_df.get_derivatives(subject, all_deriv))
            )
        ]

        # Update the target_filename list by checking if that derivative contain respective target suffix
        target_filename, list_derivative_path = self.get_most_relevant_target_filename(subject, target_suffix, bids_df_derivatives)

        # Filter BIDS_DF for sessions not nan.
        # If found same session, use that.
        # if not sort path (as session is already ascending) and use the EARLIEST session data.

        # Go through each derivative
        for derivative_path in list_derivative_path:
            if not (self.roi_params[ROIParamsKW.SUFFIX] is None) and self.roi_params[ROIParamsKW.SUFFIX] in derivative_path:
                roi_filename = [derivative_path]

        # Multiline check for valid target_filename and valid roi_filename before proceeding
        missing_target_filename: bool = not any(target_filename)

        require_roi_suffix: bool = not self.roi_params[ROIParamsKW.SUFFIX] is None
        roi_file_is_empty: bool = roi_filename is None
        missing_roi_filename: bool = require_roi_suffix and roi_file_is_empty

        # Early return when missing data.
        if missing_target_filename:
            logger.warning(f"Unable to locate either target file name for {subject}.")
            return
        elif missing_roi_filename:
            logger.warning(f"Unable to locate ROI file name for {subject}.")
            return

        # Obtain subject data frame first record as metadata.
        metadata: dict = df_sub.to_dict(orient='records')[0]

        # Manually update its contrast
        metadata[MetadataKW.CONTRAST] = contrast

        if len(bounding_box_dict):
            # Take only one bounding box for cropping
            metadata[MetadataKW.BOUNDING_BOX] = bounding_box_dict[str(df_sub[BidsDataFrameKW.PATH].values[0])][0]

        # Early return if not all meta data are present in MRI parameters.
        if metadata_choice == MetadataKW.MRI_PARAMS:
            if not all(
                    [imed_film.check_isMRIparam(m, metadata, subject, self.metadata) for m in self.metadata.keys()]
            ):
                logger.warning(f"Not all meta data are present for {subject}")
                return
        elif metadata_choice and metadata_choice != MetadataKW.CONTRASTS and metadata_choice is not None:
            self.create_metadata_dict(metadata, metadata_choice, df_sub, bids_df)

        return df_sub, roi_filename, target_filename, metadata

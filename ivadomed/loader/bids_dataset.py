from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm

from ivadomed.loader import film as imed_film
from ivadomed.loader.mri2d_segmentation_dataset import MRI2DSegmentationDataset
from ivadomed.loader.tools.slice_filter import SliceFilter
from ivadomed.object_detection import utils as imed_obj_detect
from ivadomed.loader.tools.bids_dataframe import BidsDataframe
from loguru import logger
from ivadomed.keywords import ContrastParamsKW, SubjectDictKW, BidsDataFrameKW, ModelParamsKW, MetadataParamsKW, \
    MetadataChoiceKW, ROIParamsKW


def initialize_multichannel_subjects_dict_with_none(multichannel_subjects: dict, known_unique_session_list: np.ndarray,
                                                    num_contrast: int, subject_ids: list):
    """
    A utility function to initialize the MultiChannel Subjects Dictionary based on the number of contrasts with the
    correct number of Nones as place holder

    Args:
        multichannel_subjects:
        known_unique_session_list:
        num_contrast:
        subject_ids:

    """
    from ivadomed.keywords import SubjectDictKW

    if known_unique_session_list.size != 0:
        # For each subject, update the dict with the right number of None or None filled list.
        for subject in subject_ids:
            multichannel_subjects.update(
                {
                    subject: {
                        SubjectDictKW.ABSOLUTE_PATHS: [None] * num_contrast * len(known_unique_session_list),
                        SubjectDictKW.DERIV_PATH: None,
                        SubjectDictKW.ROI_FILENAME: None,
                        SubjectDictKW.METADATA: [None] * num_contrast * len(known_unique_session_list)
                    }
                }
            )
    else:
        # For each subject, update the dict with the right number of None or None filled list.
        for subject in subject_ids:
            multichannel_subjects.update(
                {
                    subject: {
                        SubjectDictKW.ABSOLUTE_PATHS: [None] * num_contrast,
                        SubjectDictKW.DERIV_PATH: None,
                        SubjectDictKW.ROI_FILENAME: None,
                        SubjectDictKW.METADATA: [None] * num_contrast
                    }
                }
            )
    return multichannel_subjects


def initialize_contrast_counter(contrast_params: dict) -> dict:
    """
    Initialize a special dict that keeps track of the contrasts for balance reason.
    Args:
        contrast_params (dict): Configuration string key about the contrast parameters

    Returns:
        contrast_counter (dict)
    """
    contrast_counter = {}
    for contrast in contrast_params[ContrastParamsKW.BALANCE].keys():
        contrast_counter.update({contrast: 0})

    return contrast_counter


def get_target_filename(target_suffix, target_filename, derivative):
    for idx, suffix_list in enumerate(target_suffix):
        # If suffix_list is a string, then only one rater annotation per class is available.
        # Otherwise, multiple raters segmented the same class.
        if isinstance(suffix_list, list):
            for suffix in suffix_list:
                if suffix in derivative:
                    target_filename[idx].append(derivative)
        elif suffix_list in derivative:
            target_filename[idx] = derivative


def get_unique_session_list(df_subjects: pd.DataFrame) -> np.ndarray:
    """
    Scavenge through the subjects dataframe and check its filenames for patterns about session information.
    Args:
        df_subjects (pd.DataFrame): DataFrame containing all subjects' file information etc

    Returns:
        list_unique_sessions (np.ndarray): all the unique sessions

    """
    # Gather the sessions first by looking through the panda dataframe.
    list_sessions = []
    for filename in df_subjects['filename']:
        if "ses-" in filename:
            list_sessions.append(filename.split("_")[1])

    # Unique sessions to be returned.
    list_unique_sessions = np.unique(list_sessions)

    return list_unique_sessions


def create_metadata_dict(metadata_choice: str, df_sub: pd.DataFrame, bids_df: pd.DataFrame, metadata: dict):
    """
    Create the dictionary of metadata
    Args:
        metadata_choice:str
        df_sub:
        bids_df:
        metadata:

    Returns:

    """

    if metadata_choice not in df_sub.columns:
        raise ValueError(f"The following metadata cannot be found: {metadata_choice}.Invalid metadata choice.")
    # add custom data to metadata
    metadata[metadata_choice] = df_sub[metadata_choice].values[0]

    # Create metadata dict for OHE
    data_lst = sorted(set(bids_df.df[metadata_choice].dropna().values))

    metadata_dict = {}
    for idx, data in enumerate(data_lst):
        metadata_dict[data] = idx

    metadata['metadata_dict'] = metadata_dict


def fill_multichannel_dict(multichannel_subjects, subject, idx_dict, sess_dict, df_sub,
                           roi_filename, target_filename, metadata):
    """
    Update the multi channel dictionary.
    Args:
        multichannel_subjects:
        subject:
        idx_dict:
        sess_dict:
        df_sub:
        roi_filename:
        target_filename:
        metadata:

    Returns:

    """

    if "ses-" not in subject:
        idx = idx_dict[df_sub['suffix'].values[0]]
        file_session = []
    else:
        file_session = subject.split("_")[1]
        idx = (len(sess_dict) - 1) * sess_dict[file_session] + idx_dict[df_sub['suffix'].values[0]]

    subj_id = subject.split('.')[0].split('_')[0]
    multichannel_subjects[subj_id][SubjectDictKW.ABSOLUTE_PATHS][idx] = df_sub['path'].values[0]
    multichannel_subjects[subj_id][SubjectDictKW.DERIV_PATH] = target_filename
    multichannel_subjects[subj_id][SubjectDictKW.METADATA][idx] = metadata
    if roi_filename:
        multichannel_subjects[subj_id][SubjectDictKW.ROI_FILENAME] = roi_filename
    return multichannel_subjects


def initialize_contrast_balance_number_of_subjects(contrast_params: dict, df_subjects: pd.DataFrame):
    """
    Create a dictionary with the number of subjects for each contrast of contrast_balance
    Args:
        contrast_params (dict): config contrast parameters dictionary
        df_subjects: the dataframe containing all subjects.

    Returns:

    """
    tot = {}
    for contrast in contrast_params[ContrastParamsKW.BALANCE].keys():
        tot.update(
            {
                contrast: df_subjects[BidsDataFrameKW.SUFFIX].str.fullmatch(contrast).value_counts()[True]
            }
        )
    return tot


class BidsDataset(MRI2DSegmentationDataset):
    """ BIDS specific dataset loader.

    Args:
        bids_df (BidsDataframe): Object containing dataframe with all BIDS image files and their metadata.

        subject_file_lst (list): Subject filenames list.

        target_suffix (list): List of suffixes for target masks.

        contrast_params (dict): Contains image contrasts related parameters.

        model_params (dict): Dictionary containing model parameters.

        slice_axis (int): Indicates the axis used to extract 2D slices from 3D nifti files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D png/tif/jpg files use default "axial": 2.

        cache (bool): If the data should be cached in memory or not.

        transform (list): Transformation list (length 2) composed of preprocessing transforms (Compose) and transforms
            to apply during training (Compose).

        metadata_choice (str): Choice between "mri_params", "contrasts", the name of a column from the
            participants.tsv file, None or False, related to FiLM.

        slice_filter_fn (SliceFilter): Class that filters slices according to their content.

        roi_params (dict): Dictionary containing parameters related to ROI image processing.

        multichannel (bool): If True, the input contrasts are combined as input channels for the model. Otherwise, each
            contrast is processed individually (ie different sample / tensor).

        object_detection_params (dict): Object detection parameters.

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

    def __init__(self, bids_df: BidsDataframe, subject_file_lst: list, target_suffix: list, contrast_params: dict,
                 model_params: dict, slice_axis: int = 2, cache: bool = True, transform: list = None,
                 metadata_choice: str = False, slice_filter_fn: SliceFilter = None, roi_params: dict = None,
                 multichannel: bool = False, object_detection_params: dict = None, task: str = "segmentation",
                 soft_gt: bool = False, is_input_dropout: bool = False):

        self.roi_params = roi_params if roi_params is not None else {"suffix": None, "slice_filter_roi": None}
        self.soft_gt = soft_gt
        self.filename_pairs = []
        if metadata_choice == 'mri_params':
            self.metadata = {"FlipAngle": [], "RepetitionTime": [],
                             "EchoTime": [], "Manufacturer": []}

        # Sort subject_file_lst and create a sub-dataframe from bids_df containing only subjects from subject_file_lst
        subject_file_lst: list = sorted(subject_file_lst)
        df_subjects: pd.DataFrame = bids_df.df[bids_df.df[BidsDataFrameKW.FILENAME].isin(subject_file_lst)]

        # Backward compatibility for subject_file_lst containing participant_ids instead of filenames
        if df_subjects.empty:
            df_subjects = bids_df.df[
                bids_df.df[BidsDataFrameKW.PARTICIPANT_ID].isin(subject_file_lst)
            ]
            subject_file_lst = sorted(df_subjects[BidsDataFrameKW.FILENAME].to_list())

        tot = initialize_contrast_balance_number_of_subjects(contrast_params, df_subjects)

        # Create a counter that helps to balance the contrasts
        c = initialize_contrast_counter(contrast_params)

        # Get a list of subject_ids for multichannel_subjects (prefix filename without modality suffix and extension)
        subject_ids: list = []
        for subject in subject_file_lst:
            subject_ids.append(subject.split('.')[0].split('_')[0])
        subject_ids = sorted(list(set(subject_ids)))

        # Create multichannel_subjects dictionary for each subject_id
        multichannel_subjects: dict = {}
        idx_dict: dict = {}

        # Storing session information in dict
        sess_dict: dict = {}

        # Multi channel flag
        if multichannel:

            logger.debug("Multichannel BIDSDataset detected.")

            # Gauge the number of contrasts specified form the configuration file
            num_contrast = len(contrast_params[ContrastParamsKW.CONTRAST_LIST])

            # Derive the session information FROM FILEs by examining the dataframe of subjects.
            known_unique_session_list: np.ndarray = get_unique_session_list(df_subjects)

            # Prefill the the contrast dict with the right number of Nones
            for contrast_index, contrast in enumerate(contrast_params[ContrastParamsKW.CONTRAST_LIST]):
                idx_dict[contrast] = contrast_index

            # Prefill the the session dict with the right number of Nones
            for session_index, session in enumerate(known_unique_session_list):
                sess_dict[session] = session_index

            # Initialize the multichannel subjects dict with none
            initialize_multichannel_subjects_dict_with_none(multichannel_subjects, known_unique_session_list,
                                                            num_contrast, subject_ids)

        # Get all subjects path from bids_df for bounding box
        get_all_subj_path = bids_df.df[bids_df.df[BidsDataFrameKW.FILENAME]
            .str.contains('|'.join(bids_df.get_subject_fnames()))]['path'].to_list()

        # Load bounding box from list of path
        bounding_box_dict = imed_obj_detect.load_bounding_boxes(object_detection_params,
                                                                get_all_subj_path,
                                                                slice_axis,
                                                                contrast_params[ContrastParamsKW.CONTRAST_LIST])

        # Get all derivatives filenames from bids_df
        all_deriv: list = bids_df.get_deriv_fnames()

        # Create filename_pairs
        for subject in tqdm(subject_file_lst, desc="Loading dataset"):
            df_sub, roi_filename, target_filename, metadata = self.create_filename_pair(subject,
                                                                                        c, tot, df_subjects,
                                                                                        contrast_params, target_suffix,
                                                                                        all_deriv, bids_df,
                                                                                        bounding_box_dict,
                                                                                        metadata_choice)
            # Fill multichannel dictionary
            # subj_id is the filename without modality suffix and extension
            if multichannel:
                multichannel_subjects = fill_multichannel_dict(multichannel_subjects, subject, idx_dict, sess_dict,
                                                                    df_sub, roi_filename, target_filename, metadata)
            else:
                self.filename_pairs.append(
                    (
                        [df_sub['path'].values[0]],
                        target_filename,
                        roi_filename,
                        [metadata]
                    )
                )

        if multichannel:
            for subject in multichannel_subjects.values():
                if None not in subject[SubjectDictKW.ABSOLUTE_PATHS]:
                    self.filename_pairs.append(
                        (
                            subject[SubjectDictKW.ABSOLUTE_PATHS],
                            subject[SubjectDictKW.DERIV_PATH],
                            subject[SubjectDictKW.ROI_FILENAME],
                            subject[SubjectDictKW.METADATA]
                        )
                    )

        if not self.filename_pairs:
            raise Exception('No subjects were selected - check selection of parameters on config.json (e.g. center '
                            'selected + target_suffix)')

        length = model_params[ModelParamsKW.LENGTH_2D] if ModelParamsKW.LENGTH_2D in model_params else []
        stride = model_params[ModelParamsKW.STRIDE_2D] if ModelParamsKW.STRIDE_2D in model_params else []

        super().__init__(self.filename_pairs, length, stride, slice_axis, cache, transform, slice_filter_fn, task,
                         self.roi_params,
                         self.soft_gt, is_input_dropout)

    def create_filename_pair(self, subject, c, tot, df_subjects, contrast_params,
                             target_suffix, all_deriv, bids_df, bounding_box_dict, metadata_choice):
        """
        Create the file name pairs
        Args:
            subject:
            c:
            tot:
            df_subjects:
            contrast_params:
            target_suffix:
            all_deriv:
            bids_df:
            bounding_box_dict:
            metadata_choice:

        Returns:

        """

        df_sub = df_subjects.loc[df_subjects['filename'] == subject]

        # Training & Validation: do not consider the contrasts over the threshold contained in contrast_balance
        contrast = df_sub['suffix'].values[0]
        if contrast in (contrast_params[ContrastParamsKW.BALANCE].keys()):
            c[contrast] = c[contrast] + 1
            if c[contrast] / tot[contrast] > contrast_params[ContrastParamsKW.BALANCE][contrast]:
                return
        if isinstance(target_suffix[0], str):
            target_filename, roi_filename = [None] * len(target_suffix), None
        else:
            target_filename, roi_filename = [[] for _ in range(len(target_suffix))], None

        derivatives = bids_df.df[bids_df.df[BidsDataFrameKW.FILENAME]
            .str.contains('|'.join(bids_df.get_derivatives(subject, all_deriv)))]['path'].to_list()

        for derivative in derivatives:
            get_target_filename(target_suffix, target_filename, derivative)
            if not (self.roi_params[ROIParamsKW.SUFFIX] is None) and \
                    self.roi_params[ROIParamsKW.SUFFIX] in derivative:
                roi_filename = [derivative]

        if (not any(target_filename)) or (
                not (self.roi_params[ROIParamsKW.SUFFIX] is None) and (roi_filename is None)):
            return

        metadata = df_sub.to_dict(orient='records')[0]
        metadata[MetadataParamsKW.CONTRAST] = contrast

        if len(bounding_box_dict):
            # Take only one bounding box for cropping
            metadata[MetadataParamsKW.BOUNDING_BOX] = bounding_box_dict[str(df_sub['path'].values[0])][0]

        if metadata_choice == MetadataChoiceKW.MRI_PARAMS:
            if not all([imed_film.check_isMRIparam(m, metadata, subject, self.metadata) for m in
                        self.metadata.keys()]):
                return

        elif metadata_choice and metadata_choice != MetadataChoiceKW.CONTRASTS and metadata_choice is not None:
            create_metadata_dict(metadata_choice, df_sub, bids_df, metadata)

        return df_sub, roi_filename, target_filename, metadata

from __future__ import annotations
from pathlib import Path
import os
from typing import List
from ivadomed.keywords import FileStemKW
from loguru import logger


class SubjectAggregation:
    def __init__(self):
        self.subject: str = ""
        self.acq: str = ""
        self.session: str = ""
        self.run: str = ""

        self.sample: str = ""
        self.modality: str = ""
        self.derivative: str = ""
        self.location: str = ""
        self.extensions: str = ""
        self.stem: str = ""  # The part before ALL the file extensions.
        self.contrast_params: str = ""
        self.list_stem: List[str] = []

    @staticmethod
    def instantiate(file_name: str) -> SubjectAggregation:
        """
        Instantiate a Subjection Aggregation
        Args:
            file_name:

        Returns:

        """
        path_filename = Path(file_name)

        subject_data = SubjectAggregation()

        # Everything before
        subject_data.extensions = path_filename.suffixes

        # Strip out all extensions:
        # stem is the aggregation that contains subject/session/acq/sample/modality etc
        subject_data.stem = os.path.splitext(path_filename.name)[0]

        subject_data.list_stem = subject_data.stem.split("_")

        if len(subject_data.list_stem) < 2:
            logger.critical("Problematic file name detected. Does not have expected number of underscore separation:"
                            f"{file_name}")
        elif len(subject_data.list_stem) == 2:
            SubjectAggregation.instantiate_raw_data(file_name)

        elif len(subject_data.list_stem) > 2:

            # Go through each stem in the list
            for stem in subject_data.list_stem:
                stem = stem.lower()

                # Subject Stem
                if any(subject_keyword in stem for subject_keyword in FileStemKW.list_subject):
                    subject_data.subject = stem

                # Session Stem
                elif any(session_keyword in stem for session_keyword in FileStemKW.list_session):
                    subject_data.session = stem

                # Acquisition Stem
                elif any(acquisition_keyword in stem for acquisition_keyword in FileStemKW.list_acquisition):
                    subject_data.acq = stem

                #  Sample Stem
                elif any(sample_keyword in stem for sample_keyword in FileStemKW.list_sample):
                    subject_data.sample = stem

                # Run Stem
                elif any(run_keyword in stem for run_keyword in FileStemKW.list_run):
                    subject_data.run = stem

                # Modality Stem
                elif any(contrast_keyword in stem for contrast_keyword in FileStemKW.list_contrasts):
                    subject_data.modality = stem

                # Derivative Stem
                elif any(derivative_keyword in stem for derivative_keyword in FileStemKW.list_derivative):
                    subject_data.derivative = stem

                # Location Stem
                elif any(location_keyword in stem for location_keyword in FileStemKW.list_location):
                    subject_data.location = stem

                # Stats Stem
                elif any(stats_keyword in stem for stats_keyword in FileStemKW.list_stats):
                    subject_data.stats = stem

                # Unanticipated Stem
                else:
                    logger.warning(f"Unrecognized filename part stem: {stem}")

            # Must have these two things to be considered valid.
        if not subject_data.subject:
            raise ValueError(f"Subject related FileStem keyword is not found in the file name: {file_name}")
        if not subject_data.modality:
            raise ValueError(f"Modality related FileStem keyword is not found in the file name: {file_name}")

        return subject_data

    @staticmethod
    def instantiate_raw_data(file_name: str) -> SubjectAggregation:
        """
        Instantiate a Subjection Aggregation
        Args:
            file_name:

        Returns:

        """
        path_filename = Path(file_name)

        subject_data = SubjectAggregation()

        # Everything before
        subject_data.extensions = path_filename.suffixes

        # Strip out all extensions:
        # stem is the aggregation that contains subject/session/acq/sample/modality etc
        subject_data.stem = os.path.splitext(path_filename.name)[0]

        subject_data.list_stem = subject_data.stem.split("_")

        # Should really have two parts separated: subject and contrast
        if len(subject_data.list_stem) != 2:
            logger.warning(f"Unexpected file name with potentially more than a single '_' in side the file name: {file_name}")

        # Go through each stem in the list
        for stem in subject_data.list_stem:
            stem = stem.lower()

            # Subject Stem
            if stem in FileStemKW.list_subject:
                subject_data.subject = stem
            # Acquisition Stem
            elif stem in FileStemKW.list_contrasts:
                subject_data.modality = stem
            # Unanticipated Stem
            else:
                logger.warning(f"Unrecognized filename part stem: {stem} in {subject_data.stem}")

        # Must have these two things to be considered valid.
        if not subject_data.subject:
            raise ValueError(f"Subject related FileStem keyword is not found in the file name: {file_name}")
        if not subject_data.modality:
            raise ValueError(f"Modality related FileStem keyword is not found in the file name: {file_name}")

        return subject_data

    @staticmethod
    def instantiate_derivatives(file_name: str) -> SubjectAggregation:
        """
        Instantiate a SubjectAggregation on a typical regular raw data input file and return it based on the file name provided.
        Args:
            file_name:

        Returns:

        """
        path_filename = Path(file_name)

        subject_data = SubjectAggregation()

        # Everything before
        subject_data.extensions = path_filename.suffixes

        # Strip out all extensions:
        # stem is the aggregation that contains subject/session/acq/sample/modality etc
        subject_data.stem = os.path.splitext(path_filename.name)[0]
        subject_data.stem = os.path.splitext(path_filename.name)[0]

        subject_data.list_stem = subject_data.stem.split("_")

        # Subject is always first.
        # self.subject = self.list_stem[0]

        # Go through each stem in the list
        for stem in subject_data.list_stem:
            stem = stem.lower()

            # Subject Stem
            if stem in FileStemKW.list_subject:
                subject_data.subject = stem
            # Acquisition Stem
            elif stem in FileStemKW.list_acquisition:
                subject_data.acq = stem
            #  Sample Stem
            elif stem in FileStemKW.list_sample:
                subject_data.sample = stem
            # Session Stem
            elif stem in FileStemKW.list_session:
                subject_data.session = stem
            # Run Stem
            elif stem in FileStemKW.list_run:
                subject_data.run = stem
            # Modality Stem
            elif stem in FileStemKW.list_contrasts:
                subject_data.modality = stem
            # Derivative Stem
            elif stem in FileStemKW.list_derivative:
                subject_data.derivative = stem
            # Location Stem
            elif stem in FileStemKW.list_location:
                subject_data.location = stem
            # Stats Stem
            elif stem in FileStemKW.list_stats:
                subject_data.stats = stem

            # Unanticipated Stem
            else:
                logger.warning(f"Unrecognized filename part stem: {stem}")

        # Must have these two things to be considered valid.
        if not subject_data.subject:
            raise ValueError(f"Subject related FileStem keyword is not found in the file name: {file_name}")
        if not subject_data.modality:
            raise ValueError(f"Modality related FileStem keyword is not found in the file name: {file_name}")

        return subject_data


# Need to test or scenarios such as:
# sub-unf01_T2w_labels-disc-manual.nii.gz
# sub-unf01_T2w_mid_heatmap3.nii.gz
# sub-unf01_T1w.json
# sub-unf01_T2w_mid.nii.gz
# sub-rat2_sample-data5_SEM.json
# sub-rat3_ses-01_sample-data9_SEM.json
# sub-rat3_ses-01_sample-data9_SEM_seg-axon-manual.png
# sub-rat2_sample-data5_SEM_seg-axon-manual.png
# sub-unf02_T1w_labels-disc-manual.nii.gz
# sub-spleen2_ct_seg-manual.nii.gz
# sub-spleen2_ct.nii.gz

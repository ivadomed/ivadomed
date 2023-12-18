from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from ivadomed.loader.generalized_loader_configuration import (
        GeneralizedLoaderConfiguration,
    )

from ivadomed.loader.bids_dataset import BidsDataset

from ivadomed.loader.files_dataset import FilesDataset
from ivadomed.loader.mri3d_subvolume_segmentation_dataset import (
    MRI3DSubVolumeSegmentationDataset,
)
from ivadomed.keywords import ModelParamsKW


class Files3DDataset(MRI3DSubVolumeSegmentationDataset):
    """
    3D Version fo the FilesDataset
    """
    def __init__(
        self,
        dict_files_pairing: dict,
        config: GeneralizedLoaderConfiguration,
    ):
        dataset = FilesDataset(dict_files_pairing, config)

        super().__init__(
            dataset.filename_pairs,
            length=config.model_params[ModelParamsKW.LENGTH_3D],
            stride=config.model_params[ModelParamsKW.STRIDE_3D],
            transform=config.transform,
            slice_axis=config.slice_axis,
            task=config.task,
            soft_gt=config.soft_gt,
            is_input_dropout=config.is_input_dropout,
            disk_cache=config.disk_cache,
        )

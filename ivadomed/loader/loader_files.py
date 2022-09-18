import copy
from loguru import logger
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader.bids3d_dataset import Bids3DDataset
from ivadomed.loader.bids_dataset import BidsDataset
from ivadomed.keywords import ROIParamsKW, TransformationKW, ModelParamsKW, ConfigKW
from ivadomed.loader.files3d_dataset import Files3DDataset
from ivadomed.loader.files_dataset import FilesDataset
from ivadomed.loader.generalized_loader_configuration import (
    GeneralizedLoaderConfiguration,
)
from ivadomed.loader.slice_filter import SliceFilter
from ivadomed.loader.patch_filter import PatchFilter


def load_fileset(
    filesets_dict: dict,
    model_params: dict,
    transforms_params: dict,
    roi_params: dict,
    slice_axis: int,
    dataset_type: str="training",
    requires_undo: bool=False,
    **kwargs,
):
    # Get loader appropriate loader according to model type:
    # Available loaders are :
    #   Files3DDataset for 3D data
    #   FilesDataset for 2D data

    # Compose transforms
    tranform_lst, _ = imed_transforms.prepare_transforms(
        copy.deepcopy(transforms_params), requires_undo
    )

    # If ROICrop is not part of the transforms, then enforce no slice filtering based on ROI data.
    if TransformationKW.ROICROP not in transforms_params:
        roi_params[ROIParamsKW.SLICE_FILTER_ROI] = None

    config = GeneralizedLoaderConfiguration(
        model_params=model_params,
        transform=tranform_lst,
    )

    # Generate the dataset object to allow data loading.
    if model_params[ModelParamsKW.NAME] == ConfigKW.MODIFIED_3D_UNET or (
        ModelParamsKW.IS_2D in model_params and not model_params[ModelParamsKW.IS_2D]
    ):
        dataset = Files3DDataset(filesets_dict, config)
    else:
        # Task selection
        dataset = FilesDataset(filesets_dict, config)

    # 3D Path
    if model_params[ModelParamsKW.NAME] == ConfigKW.MODIFIED_3D_UNET:
        logger.info(
            f"Loaded {len(dataset)} volumes of shape {dataset.length} for the {dataset_type} set."
        )
    # ??? Patches Path
    elif model_params[ModelParamsKW.NAME] != ConfigKW.HEMIS_UNET and dataset.length:
        logger.info(
            f"Loaded {len(dataset)} {slice_axis} patches of shape {dataset.length} for the {dataset_type} set."
        )
    # 2D Slices Path
    else:
        logger.info(
            f"Loaded {len(dataset)} {slice_axis} slices for the { dataset_type} set."
        )

    return dataset

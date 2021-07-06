import os
import nibabel as nib
import numpy as np
import onnxruntime
import torch
import joblib
from typing import List

from loguru import logger
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor

from ivadomed.loader.mri3d_subvolume_segmentation_dataset import MRI3DSubVolumeSegmentationDataset
from ivadomed.loader.mri2d_segmentation_dataset import MRI2DSegmentationDataset
from ivadomed.transforms import UndoCompose
from ivadomed import config_manager as imed_config_manager
from ivadomed import models as imed_models
from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed.loader import utils as imed_loader_utils, film as imed_film
from ivadomed.object_detection import utils as imed_obj_detect
from ivadomed import utils as imed_utils
from ivadomed import training as imed_training



def onnx_inference(model_path: str, inputs: tensor) -> tensor:
    """Run ONNX inference

    Args:
        model_path (str): Path to the ONNX model.
        inputs (Tensor): Batch of input image.

    Returns:
        Tensor: Network output.
    """
    inputs = np.array(inputs.cpu())
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    return torch.tensor(ort_outs[0])


def get_preds(context: dict, fname_model: str, model_params: dict, gpu_id: int, batch: dict) -> tensor:
    """Returns the predictions from the given model.

    Args:
        context (dict): configuration dict.
        fname_model (str): name of file containing model.
        model_params (dict): dictionary containing model parameters.
        gpu_id (int): Number representing gpu number if available. Currently does NOT support multiple GPU segmentation.
        batch (dict): dictionary containing input, gt and metadata

    Returns:
        tensor: predictions from the model.
    """
    # Define device
    cuda_available, device = imed_utils.define_device(gpu_id)

    with torch.no_grad():

        # Load the Input
        img = imed_utils.cuda(batch['input'], cuda_available=cuda_available)

        # Load the PyTorch model and evaluate if model files exist.
        if fname_model.lower().endswith('.pt'):
            logger.debug(f"PyTorch model detected at: {fname_model}")
            logger.debug(f"Loading model from: {fname_model}")
            model = torch.load(fname_model, map_location=device)
            # Inference time
            logger.debug(f"Evaluating model: {fname_model}")
            model.eval()

            # Films/Hemis based prediction require meta data load
            if ('FiLMedUnet' in context and context['FiLMedUnet']['applied']) or \
                    ('HeMISUnet' in context and context['HeMISUnet']['applied']):
                # Load meta data before prediction
                metadata = imed_training.get_metadata(batch["input_metadata"], model_params)
                preds = model(img, metadata)
            else:
                preds = model(img)
        # Otherwise, Onnex Inference (PyTorch can't load .onnx)
        else:
            logger.debug(f"Likely ONNX model detected at: {fname_model}")
            logger.debug(f"Conduct ONNX model inference... ")
            preds = onnx_inference(fname_model, img)

        logger.debug("Sending predictions to CPU")
        # Move prediction to CPU
        preds = preds.cpu()

    return preds


def get_onehotencoder(context: dict, folder_model: str, options: dict, ds: Dataset) -> dict:
    """Returns one hot encoder which is needed to update the model parameters when FiLMedUnet is applied.

    Args:
        context (dict): configuration dict.
        folder_model (str): foldername which contains trained model and its configuration file.
        options (dict): contains postprocessing steps and prior filename containing processing information
        ds (Dataset): dataset used for the segmentation.

    Returns:
        dict: onehotencoder used in the model params.
    """
    metadata_dict = joblib.load(os.path.join(folder_model, 'metadata_dict.joblib'))
    for idx in ds.indexes:
        for i in range(len(idx)):
            idx[i]['input_metadata'][0][context['FiLMedUnet']['metadata']] = options['metadata']
            idx[i]['input_metadata'][0]['metadata_dict'] = metadata_dict

    ds = imed_film.normalize_metadata(ds, None, context["debugging"], context['FiLMedUnet']['metadata'])

    return joblib.load(os.path.join(folder_model, 'one_hot_encoder.joblib'))


def pred_to_nib(data_lst: List[np.ndarray], z_lst: List[int], fname_ref: str, fname_out: str, slice_axis: int,
                debug: bool = False, kernel_dim: str='2d', bin_thr: float=0.5, discard_noise: bool = True,
                postprocessing: dict = None) -> nib.Nifti1Image:
    """Save the network predictions as nibabel object.

    Based on the header of `fname_ref` image, it creates a nibabel object from the Network predictions (`data_lst`).

    Args:
        data_lst (list of np arrays): Predictions, either 2D slices either 3D patches.
        z_lst (list of ints): Slice indexes to reconstruct a 3D volume for 2D slices.
        fname_ref (str): Filename of the input image: its header is copied to the output nibabel object.
        fname_out (str): If not None, then the generated nibabel object is saved with this filename.
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
        debug (bool): If True, extended verbosity and intermediate outputs.
        kernel_dim (str): Indicates whether the predictions were done on 2D or 3D patches. Choices: '2d', '3d'.
        bin_thr (float): If positive, then the segmentation is binarized with this given threshold. Otherwise, a soft
            segmentation is output.
        discard_noise (bool): If True, predictions that are lower than 0.01 are set to zero.
        postprocessing (dict): Contains postprocessing steps to be applied.

    Returns:
        nibabel.Nifti1Image: NiBabel object containing the Network prediction.
    """

    # Check fname_ref extention and update path if not NifTI
    fname_ref = imed_loader_utils.update_filename_to_nifti(fname_ref)

    # Load reference nibabel object
    nib_ref = nib.load(fname_ref)
    nib_ref_can = nib.as_closest_canonical(nib_ref)

    if kernel_dim == '2d':
        # complete missing z with zeros
        tmp_lst = []
        for z in range(nib_ref_can.header.get_data_shape()[slice_axis]):
            if z not in z_lst:
                tmp_lst.append(np.zeros(data_lst[0].shape))
            else:
                tmp_lst.append(data_lst[z_lst.index(z)])

        if debug:
            logger.debug(f"Len {len(tmp_lst)}")
            for arr in tmp_lst:
                logger.debug(f"Shape element lst {arr.shape}")

        # create data and stack on depth dimension
        arr_pred_ref_space = np.stack(tmp_lst, axis=-1)

    else:
        arr_pred_ref_space = data_lst[0]

    n_channel = arr_pred_ref_space.shape[0]
    oriented_volumes = []
    if len(arr_pred_ref_space.shape) == 4:
        for i in range(n_channel):
            oriented_volumes.append(
                imed_loader_utils.reorient_image(arr_pred_ref_space[i, ], slice_axis, nib_ref, nib_ref_can))
        # transpose to locate the channel dimension at the end to properly see image on viewer
        arr_pred_ref_space = np.asarray(oriented_volumes).transpose((1, 2, 3, 0))
    else:
        arr_pred_ref_space = imed_loader_utils.reorient_image(arr_pred_ref_space, slice_axis, nib_ref, nib_ref_can)

    if bin_thr >= 0:
        arr_pred_ref_space = imed_postpro.threshold_predictions(arr_pred_ref_space, thr=bin_thr)
    elif discard_noise:  # discard noise
        arr_pred_ref_space[arr_pred_ref_space <= 1e-2] = 0

    # create nibabel object
    if postprocessing:
        fname_prefix = fname_out.split("_pred.nii.gz")[0] if fname_out is not None else None
        postpro = imed_postpro.Postprocessing(postprocessing,
                                              arr_pred_ref_space,
                                              nib_ref.header['pixdim'][1:4],
                                              fname_prefix)
        arr_pred_ref_space = postpro.apply()

    # Here we prefer to copy the header (rather than just the affine matrix), in order to preserve the qform_code.
    # See: https://github.com/ivadomed/ivadomed/issues/711
    nib_pred = nib.Nifti1Image(
        dataobj=arr_pred_ref_space,
        affine=None,
        header=nib_ref.header.copy()
    )
    # save as NifTI file
    if fname_out is not None:
        nib.save(nib_pred, fname_out)

    return nib_pred


def process_transformations(context: dict, fname_roi: str, fname_prior: str, metadata: dict, slice_axis: int,
                            fname_images: list) -> dict:
    """Sets the transformation based on context parameters. When ROI is not provided center-cropping is applied.
       If there is an object_detection_path, then we modify the metadata to store transformation data.

    Args:
        context (dict): configuration dictionary.
        fname_roi (str): filename containing region for cropping image prior to segmentation.
        fname_prior (str): prior image filename.
        metadata (dict): metadata used in setting bounding box when we have object_detection_params.
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
        fname_images (list): list of image filenames (e.g. .nii.gz) to segment.

    Returns:
        dict: metadata.
    """
    if fname_roi is None and 'ROICrop' in context["transformation"].keys():
        logger.warning(
            "fname_roi has not been specified, then a cropping around the center of the image is "
            "performed instead of a cropping around a Region of Interest.")

        context["transformation"] = dict((key, value) if key != 'ROICrop'
                                         else ('CenterCrop', value)
                                         for (key, value) in context["transformation"].items())

    if 'object_detection_params' in context and \
            context['object_detection_params']['object_detection_path'] is not None:
        imed_obj_detect.bounding_box_prior(fname_prior, metadata, slice_axis,
                                           context['object_detection_params']['safety_factor'])
        metadata = [metadata] * len(fname_images)

    return metadata


def set_option(options: dict, postpro: dict, context: dict, key: str):
    """Generalized function that sets postprocessing option based on given list of options.
       When given key already exists in options, we initialize the key value for the postprocessing dictionary
       Otherwise, when the key is already found in the postprocessing attritute of the context, we remove it

    Args:
        options (dict): Contains postprocessing steps and prior filename (fname_prior) which is an image filename.
        postpro (dict): postprocessing settings.
        context (dict): Configuration dict.
        key (str): The key of the postprocessing option we wish to set.

    Returns:
        dict: postprocessing settings.
    """
    if options[key]:
        postpro[key] = {}
    # Remove key in context if value set to 0
    elif key in context['postprocessing']:
        del context['postprocessing'][key]


def set_postprocessing_options(options: dict, context: dict):
    """Updates the postprocessing options based on existing settings found in options.

    Args:
        options (dict): Contains postprocessing steps and prior filename (fname_prior) which is an image filename.
        context (dict): Configuration dict.
    """
    postpro = {}

    if 'binarize_prediction' in options and options['binarize_prediction']:
        postpro['binarize_prediction'] = {"thr": options['binarize_prediction']}

    if 'keep_largest' in options and options['keep_largest'] is not None:
        set_option(options, postpro, context, 'keep_largest')

    if 'fill_holes' in options and options['fill_holes'] is not None:
        set_option(options, postpro, context, 'fill_holes')

    if 'remove_small' in options and options['remove_small'] and \
            ('mm' in options['remove_small'][-1] or 'vox' in options['remove_small'][-1]):
        unit = 'mm3' if 'mm3' in options['remove_small'][-1] else 'vox'
        thr = [int(t.replace(unit, "")) for t in options['remove_small']]
        postpro['remove_small'] = {"unit": unit, "thr": thr}

    context['postprocessing'].update(postpro)


def segment_volume(folder_model: str, fname_images: list, gpu_id: int = 0, options: dict = None):
    """Segment an image.

    Segment an image (`fname_image`) using a pre-trained model (`folder_model`). If provided, a region of interest
    (`fname_roi`) is used to crop the image prior to segment it.

    Args:
        folder_model (str): foldername which contains
            (1) the model ('folder_model/folder_model.pt') to use
            (2) its configuration file ('folder_model/folder_model.json') used for the training,
            see https://github.com/neuropoly/ivadomed/wiki/configuration-file
        fname_images (list): list of image filenames (e.g. .nii.gz) to segment. Multichannel models require multiple
            images to segment, e.i., len(fname_images) > 1.
        gpu_id (int): Number representing gpu number if available. Currently does NOT support multiple GPU segmentation.
        options (dict): Contains postprocessing steps and prior filename (fname_prior) which is an image filename
            (e.g., .nii.gz) containing processing information (e.i., spinal cord segmentation, spinal location or MS
            lesion classification)
            e.g., spinal cord centerline, used to crop the image prior to segment it if provided.
            The segmentation is not performed on the slices that are empty in this image.

    Returns:
        list: List of nibabel objects containing the soft segmentation(s), one per prediction class.
        list: List of target suffix associated with each prediction in `pred_list`

    """

    # Check if model folder exists and get filenames to be stored as string
    fname_model: str
    fname_model_metadata: str
    fname_model, fname_model_metadata = imed_models.get_model_filenames(folder_model)

    # Load model training config
    context = imed_config_manager.ConfigurationManager(fname_model_metadata).get_config()

    postpro_list = ['binarize_prediction', 'keep_largest', ' fill_holes', 'remove_small']
    if options is not None and any(pp in options for pp in postpro_list):
        set_postprocessing_options(options, context)

    # LOADER
    loader_params = context["loader_parameters"]
    slice_axis = imed_utils.AXIS_DCT[loader_params['slice_axis']]
    metadata = {}
    fname_roi = None
    fname_prior = options['fname_prior'] if (options is not None) and ('fname_prior' in options) else None
    if fname_prior is not None:
        if 'roi_params' in loader_params and loader_params['roi_params']['suffix'] is not None:
            fname_roi = fname_prior
        # TRANSFORMATIONS
        metadata = process_transformations(context, fname_roi, fname_prior, metadata, slice_axis, fname_images)

    # Compose transforms
    _, _, transform_test_params = imed_transforms.get_subdatasets_transforms(context["transformation"])

    tranform_lst, undo_transforms = imed_transforms.prepare_transforms(transform_test_params)

    # Force filter_empty_mask to False if fname_roi = None
    if fname_roi is None and 'filter_empty_mask' in loader_params["slice_filter_params"]:
        logger.warning("fname_roi has not been specified, then the entire volume is processed.")
        loader_params["slice_filter_params"]["filter_empty_mask"] = False

    # TODO: Add PixelSize from options to filename_pairs metadata for microscopy inference (issue #306)
    filename_pairs = [(fname_images, None, fname_roi, metadata if isinstance(metadata, list) else [metadata])]

    kernel_3D = bool('Modified3DUNet' in context and context['Modified3DUNet']['applied']) or \
                not context['default_model']['is_2d']
    length_2D = context["default_model"]["length_2D"] if "length_2D" in context["default_model"] else []
    stride_2D = context["default_model"]["stride_2D"] if "stride_2D" in context["default_model"] else []
    is_2d_patch = bool(length_2D)

    if kernel_3D:
        ds = MRI3DSubVolumeSegmentationDataset(filename_pairs,
                                               transform=tranform_lst,
                                               length=context["Modified3DUNet"]["length_3D"],
                                               stride=context["Modified3DUNet"]["stride_3D"])
        logger.info(f"Loaded {len(ds)} {loader_params['slice_axis']} volumes of shape "
                     f"{context['Modified3DUNet']['length_3D']}.")
    else:
        ds = MRI2DSegmentationDataset(filename_pairs,
                                      length=length_2D,
                                      stride=stride_2D,
                                      slice_axis=slice_axis,
                                      cache=True,
                                      transform=tranform_lst,
                                      slice_filter_fn=imed_loader_utils.SliceFilter(
                                          **loader_params["slice_filter_params"]))
        ds.load_filenames()
        if is_2d_patch:
            logger.info(f"Loaded {len(ds)} {loader_params['slice_axis']} patches of shape {length_2D}.")
        else:
            logger.info(f"Loaded {len(ds)} {loader_params['slice_axis']} slices.")

    model_params = {}
    if 'FiLMedUnet' in context and context['FiLMedUnet']['applied']:
        onehotencoder = get_onehotencoder(context, folder_model, options, ds)
        model_params.update({"name": 'FiLMedUnet',
                             "film_onehotencoder": onehotencoder,
                             "n_metadata": len([ll for l in onehotencoder.categories_ for ll in l])})

    # Data Loader
    data_loader = DataLoader(ds, batch_size=context["training_parameters"]["batch_size"],
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=0)

    # Loop across batches
    preds_list, slice_idx_list = [], []
    last_sample_bool, weight_matrix, volume, image = False, None, None, None
    for i_batch, batch in enumerate(data_loader):
        preds = get_preds(context, fname_model, model_params, gpu_id, batch)

        # Set datatype to gt since prediction should be processed the same way as gt
        for b in batch['input_metadata']:
            for modality in b:
                modality['data_type'] = 'gt'

        # Reconstruct 3D object
        pred_list, target_list, last_sample_bool, weight_matrix, volume, image = reconstruct_3d_object(
            context, batch, undo_transforms, preds, preds_list, kernel_3D, is_2d_patch, slice_axis,
            slice_idx_list, data_loader, fname_images, i_batch, last_sample_bool, weight_matrix,
            volume, image
        )

    return pred_list, target_list


def split_classes(nib_prediction):
    """Split a 4D nibabel multi-class segmentation file in multiple 3D nibabel binary segmentation files.

    Args:
        nib_prediction (nibabelObject): 4D nibabel object.
    Returns:
        list of nibabelObject.
     """
    pred = nib_prediction.get_fdata()
    pred_list = []
    for c in range(pred.shape[-1]):
        class_pred = nib.Nifti1Image(pred[..., c].astype('float32'), None, nib_prediction.header.copy())
        pred_list.append(class_pred)
    return pred_list


def reconstruct_3d_object(context: dict, batch: dict, undo_transforms: UndoCompose, preds: torch.tensor,
                          preds_list: list, kernel_3D: bool, is_2d_patch: bool, slice_axis: int, slice_idx_list: list,
                          data_loader: DataLoader, fname_images: list, i_batch: int, last_sample_bool: bool,
                          weight_matrix: tensor, volume: tensor, image: tensor):
    """Reconstructs the 3D object from the current batch, and returns the list of predictions and
       targets.

    Args:

        context (dict): configuration dict.
        batch (dict): Dictionary containing input, gt and metadata
        undo_transforms (UndoCompose): Undo transforms so prediction match original image resolution and shape
        preds (tensor): Subvolume predictions
        preds_list (list of tensor): list of subvolume predictions.
        kernel_3D (bool): true when using 3D kernel.
        is_2d_patch (bool): True if length in default model params.
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
        slice_idx_list (list of int): list of indices for the axis slices.
        data_loader (DataLoader): DataLoader object containing batches using in object construction.
        fname_images (list): list of image filenames (e.g. .nii.gz) to segment.
        i_batch (int): index of current batch.
        last_sample_bool: : flag to indicate whether this is the last sample in the 3D volume
        weight_matrix (tensor): the weight matrix
        volume (tensor): the volume tensor that is being partially reconstructed through the loop
        image (tensor): the image tensor that is being partially reconstructed through the loop

    Returns:
        pred_list (list): list of predictions
        target_list (list): list of targets
        last_sample_bool (bool): flag to indicate whether this is the last sample in the 3D volume
        weight_matrix (tensor): the weight matrix. Must be returned as passing tensor by reference is NOT reliable.
        volume (tensor): the volume tensor that is being partially reconstructed through the loop. Must be returned
         as passing tensor by reference is NOT reliable.
        image (tensor): the vimage tensor that is being partially reconstructed through the loop. Must be returned
         as passing tensor by reference is NOT reliable.
    """
    pred_list = []
    target_list = []
    for i_slice in range(len(preds)):
        if "bounding_box" in batch['input_metadata'][i_slice][0]:
            imed_obj_detect.adjust_undo_transforms(undo_transforms.transforms, batch, i_slice)

        batch['gt_metadata'] = [[metadata[0]] * preds.shape[1] for metadata in batch['input_metadata']]
        if kernel_3D:
            preds_undo, metadata, last_sample_bool, volume, weight_matrix = \
                volume_reconstruction(batch, preds, undo_transforms, i_slice, volume, weight_matrix)
            preds_list = [np.array(preds_undo)]
        else:
            if is_2d_patch:
                # undo transformations for patch and reconstruct slice
                preds_i_undo, metadata_idx, last_patch_bool, image, weight_matrix = \
                    image_reconstruction(batch, preds, undo_transforms, i_slice, image, weight_matrix)
                # If last patch of the slice
                if last_patch_bool:
                    # Add new segmented slice to preds_list
                    preds_list.append(np.array(preds_i_undo))
                    # Store the slice index of preds_i_undo in the original 3D image
                    slice_idx_list.append(int(batch['input_metadata'][i_slice][0]['slice_index']))
            else:
                # undo transformations for slice
                preds_i_undo, metadata_idx = undo_transforms(preds[i_slice],
                                                             batch["gt_metadata"][i_slice],
                                                             data_type='gt')
                # Add new segmented slice to preds_list
                preds_list.append(np.array(preds_i_undo))
                # Store the slice index of preds_i_undo in the original 3D image
                slice_idx_list.append(int(batch['input_metadata'][i_slice][0]['slice_index']))

        # If last batch and last sample of this batch, then reconstruct 3D object
        if (i_batch == len(data_loader) - 1 and i_slice == len(batch['gt']) - 1) or last_sample_bool:
            pred_nib = pred_to_nib(data_lst=preds_list,
                                   fname_ref=fname_images[0],
                                   fname_out=None,
                                   z_lst=slice_idx_list,
                                   slice_axis=slice_axis,
                                   kernel_dim='3d' if kernel_3D else '2d',
                                   debug=False,
                                   bin_thr=-1,
                                   postprocessing=context['postprocessing'])

            pred_list = split_classes(pred_nib)
            target_list = context['loader_parameters']['target_suffix']

    return pred_list, target_list, last_sample_bool, weight_matrix, volume, image


def volume_reconstruction(batch: dict, pred: tensor, undo_transforms: UndoCompose, smp_idx: int,
                          volume: tensor = None, weight_matrix: tensor = None):
    """
    Reconstructs volume prediction from subvolumes used during training
    Args:
        batch (dict): Dictionary containing input, gt and metadata
        pred (tensor): Subvolume prediction
        undo_transforms (UndoCompose): Undo transforms so prediction match original image resolution and shap
        smp_idx (int): Batch index
        volume (tensor): Reconstructed volume
        weight_matrix (tensor): Weights containing the number of predictions for each voxel

    Returns:
        pred_undo (tensor): undone subvolume,
        metadata (dict): metadata,
        last_sample_bool (bool): boolean representing if its the last sample of the volume
        volume (tensor): representing the volume reconstructed
        weight_matrix (tensor): weight matrix
    """
    x_min, x_max, y_min, y_max, z_min, z_max = batch['input_metadata'][smp_idx][0]['coord']
    num_pred = pred[smp_idx].shape[0]

    # A boolean flag indicate whether the current volume is the VERY first subvolume of the entire 3D volume/space.
    # Formed by check if x_min/y_min/z_min are all NOT zero.
    first_sample: bool = (x_min == 0 and y_min == 0 and z_min == 0)

    # Get the Dimension
    x, y, z = batch['input_metadata'][smp_idx][0]['index_shape']

    # If this is the first sample, instantiate a ZERO tensor based on the dimension
    if first_sample:
        volume = torch.zeros((num_pred, x, y, z))
        weight_matrix = torch.zeros((num_pred, x, y, z))

    last_sample_bool = x_max == x and y_max == y and z_max == z

    # Average predictions
    volume[:, x_min:x_max, y_min:y_max, z_min:z_max] += pred[smp_idx]
    weight_matrix[:, x_min:x_max, y_min:y_max, z_min:z_max] += 1

    if last_sample_bool:
        volume /= weight_matrix

    pred_undo, metadata = undo_transforms(volume,
                                          batch['gt_metadata'][smp_idx],
                                          data_type='gt')
    return pred_undo, metadata, last_sample_bool, volume, weight_matrix


def image_reconstruction(batch: dict, pred: tensor, undo_transforms: UndoCompose, smp_idx: int,
                        image: tensor = None, weight_matrix: tensor = None):
    """
    Reconstructs image prediction from patches used during training
    Args:
        batch (dict): Dictionary containing input, gt and metadata
        pred (tensor): Patch prediction
        undo_transforms (UndoCompose): Undo transforms so prediction match original image resolution and shape
        smp_idx (int): Batch index
        image (tensor): Reconstructed image
        weight_matrix (tensor): Weights containing the number of predictions for each pixel

    Returns:
        pred_undo (tensor): undone patch,
        metadata (dict): metadata,
        last_sample_bool (bool): boolean representing if its the last patch of the image
        image (tensor): representing the image reconstructed
        weight_matrix (tensor): weight matrix
    """
    x_min, x_max, y_min, y_max = batch['input_metadata'][smp_idx][0]['coord']
    num_pred = pred[smp_idx].shape[0]

    # A boolean flag indicate whether the current patch is the VERY first patch of the entire 2D image.
    # Formed by check if x_min/y_min are all NOT zero
    first_patch: bool = (x_min == 0 and y_min == 0)

    # Get the Dimension
    x, y = batch['input_metadata'][smp_idx][0]['index_shape']

    # If this is the first sample, instantiate a ZERO tensor based on the dimension
    if first_patch:
        image = torch.zeros((num_pred, x, y))
        weight_matrix = torch.zeros((num_pred, x, y))

    last_patch_bool = x_max == x and y_max == y

    # Average predictions
    image[:, x_min:x_max, y_min:y_max] += pred[smp_idx]
    weight_matrix[:, x_min:x_max, y_min:y_max] += 1
    if last_patch_bool:
        image /= weight_matrix

    pred_undo, metadata = undo_transforms(image, batch['gt_metadata'][smp_idx], data_type='gt')
    return pred_undo, metadata, last_patch_bool, image, weight_matrix

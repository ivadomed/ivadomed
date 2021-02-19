import os
import nibabel as nib
import numpy as np
import onnxruntime
import torch
import joblib

from torch.utils.data import DataLoader
from ivadomed import config_manager as imed_config_manager
from ivadomed import models as imed_models
from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader, film as imed_film
from ivadomed.object_detection import utils as imed_obj_detect
from ivadomed import utils as imed_utils
from ivadomed import training as imed_training


def pred_to_nib(data_lst, z_lst, fname_ref, fname_out, slice_axis, debug=False, kernel_dim='2d', bin_thr=0.5,
                discard_noise=True, postprocessing=None):
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
        NibabelObject: Object containing the Network prediction.
    """
    # Load reference nibabel object
    nib_ref = nib.load(fname_ref)
    nib_ref_can = nib.as_closest_canonical(nib_ref)

    if kernel_dim == '2d':
        # complete missing z with zeros
        tmp_lst = []
        for z in range(nib_ref_can.header.get_data_shape()[slice_axis]):
            if not z in z_lst:
                tmp_lst.append(np.zeros(data_lst[0].shape))
            else:
                tmp_lst.append(data_lst[z_lst.index(z)])

        if debug:
            print("Len {}".format(len(tmp_lst)))
            for arr in tmp_lst:
                print("Shape element lst {}".format(arr.shape))

        # create data and stack on depth dimension
        arr_pred_ref_space = np.stack(tmp_lst, axis=-1)

    else:
        arr_pred_ref_space = data_lst[0]

    n_channel = arr_pred_ref_space.shape[0]
    oriented_volumes = []
    if len(arr_pred_ref_space.shape) == 4:
        for i in range(n_channel):
            oriented_volumes.append(
                imed_loader_utils.reorient_image(arr_pred_ref_space[i,], slice_axis, nib_ref, nib_ref_can))
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
    nib_pred = nib.Nifti1Image(arr_pred_ref_space, nib_ref.affine)

    # save as nifti file
    if fname_out is not None:
        nib.save(nib_pred, fname_out)

    return nib_pred


def segment_volume(folder_model, fname_images, gpu_id=0, options=None):
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
        gpu_id (int): Number representing gpu number if available.
        options (dict): Contains postprocessing steps and prior filename (fname_prior) which is an image filename
            (e.g., .nii.gz) containing processing information (e.i., spinal cord segmentation, spinal location or MS
            lesion classification)
            e.g., spinal cord centerline, used to crop the image prior to segment it if provided.
            The segmentation is not performed on the slices that are empty in this image.

    Returns:
        list: List of nibabel objects containing the soft segmentation(s), one per prediction class.
        list: List of target suffix associated with each prediction in `pred_list`

    """
    # Define device
    cuda_available = torch.cuda.is_available()
    device = torch.device("cpu") if not cuda_available else torch.device("cuda:" + str(gpu_id))

    # Check if model folder exists and get filenames
    fname_model, fname_model_metadata = imed_models.get_model_filenames(folder_model)

    # Load model training config
    context = imed_config_manager.ConfigurationManager(fname_model_metadata).get_config()

    postpro_list = ['binarize_prediction', 'keep_largest', ' fill_holes', 'remove_small']
    if options is not None and any(pp in options for pp in postpro_list):
        postpro = {}
        if 'binarize_prediction' in options and options['binarize_prediction']:
            postpro['binarize_prediction'] = {"thr": options['binarize_prediction']}
        if 'keep_largest' in options and options['keep_largest'] is not None:
            if options['keep_largest']:
                postpro['keep_largest'] = {}
            # Remove key in context if value set to 0
            elif 'keep_largest' in context['postprocessing']:
                del context['postprocessing']['keep_largest']
        if 'fill_holes' in options and options['fill_holes'] is not None:
            if options['fill_holes']:
                postpro['fill_holes'] = {}
            # Remove key in context if value set to 0
            elif 'fill_holes' in context['postprocessing']:
                del context['postprocessing']['fill_holes']
        if 'remove_small' in options and options['remove_small'] and \
                ('mm' in options['remove_small'][-1] or 'vox' in options['remove_small'][-1]):
            unit = 'mm3' if 'mm3' in options['remove_small'][-1] else 'vox'
            thr = [int(t.replace(unit, "")) for t in options['remove_small']]
            postpro['remove_small'] = {"unit": unit, "thr": thr}

        context['postprocessing'].update(postpro)

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
        # If ROI is not provided then force center cropping
        if fname_roi is None and 'ROICrop' in context["transformation"].keys():
            print(
                "\n WARNING: fname_roi has not been specified, then a cropping around the center of the image is "
                "performed instead of a cropping around a Region of Interest.")

            context["transformation"] = dict((key, value) if key != 'ROICrop'
                                             else ('CenterCrop', value)
                                             for (key, value) in context["transformation"].items())

        if 'object_detection_params' in context and \
                context['object_detection_params']['object_detection_path'] is not None:
            imed_obj_detect.bounding_box_prior(fname_prior, metadata, slice_axis,
                                               context['object_detection_params']['safety_factor'])
            metadata = [metadata] * len(fname_images)

    # Compose transforms
    _, _, transform_test_params = imed_transforms.get_subdatasets_transforms(context["transformation"])

    tranform_lst, undo_transforms = imed_transforms.prepare_transforms(transform_test_params)

    # Force filter_empty_mask to False if fname_roi = None
    if fname_roi is None and 'filter_empty_mask' in loader_params["slice_filter_params"]:
        print("\nWARNING: fname_roi has not been specified, then the entire volume is processed.")
        loader_params["slice_filter_params"]["filter_empty_mask"] = False

    filename_pairs = [(fname_images, None, fname_roi, metadata if isinstance(metadata, list) else [metadata])]

    kernel_3D = bool('Modified3DUNet' in context and context['Modified3DUNet']['applied']) or \
                not context['default_model']['is_2d']
    if kernel_3D:
        ds = imed_loader.MRI3DSubVolumeSegmentationDataset(filename_pairs,
                                                           transform=tranform_lst,
                                                           length=context["Modified3DUNet"]["length_3D"],
                                                           stride=context["Modified3DUNet"]["stride_3D"])
    else:
        ds = imed_loader.MRI2DSegmentationDataset(filename_pairs,
                                                  slice_axis=slice_axis,
                                                  cache=True,
                                                  transform=tranform_lst,
                                                  slice_filter_fn=imed_loader_utils.SliceFilter(
                                                      **loader_params["slice_filter_params"]))
        ds.load_filenames()

    if kernel_3D:
        print("\nLoaded {} {} volumes of shape {}.".format(len(ds), loader_params['slice_axis'],
                                                           context['Modified3DUNet']['length_3D']))
    else:
        print("\nLoaded {} {} slices.".format(len(ds), loader_params['slice_axis']))

    model_params = {}
    if 'FiLMedUnet' in context and context['FiLMedUnet']['applied']:
        metadata_dict = joblib.load(os.path.join(folder_model, 'metadata_dict.joblib'))
        for idx in ds.indexes:
            for i in range(len(idx)):
                idx[i]['input_metadata'][0][context['FiLMedUnet']['metadata']] = options['metadata']
                idx[i]['input_metadata'][0]['metadata_dict'] = metadata_dict

        ds = imed_film.normalize_metadata(ds, None, context["debugging"], context['FiLMedUnet']['metadata'])
        onehotencoder = joblib.load(os.path.join(folder_model, 'one_hot_encoder.joblib'))

        model_params.update({"name": 'FiLMedUnet',
                             "film_onehotencoder": onehotencoder,
                             "n_metadata": len([ll for l in onehotencoder.categories_ for ll in l])})

    # Data Loader
    data_loader = DataLoader(ds, batch_size=context["training_parameters"]["batch_size"],
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=0)

    # MODEL
    if fname_model.endswith('.pt'):
        model = torch.load(fname_model, map_location=device)
        # Inference time
        model.eval()

    # Loop across batches
    preds_list, slice_idx_list = [], []
    last_sample_bool, volume, weight_matrix = False, None, None
    for i_batch, batch in enumerate(data_loader):
        with torch.no_grad():
            img = imed_utils.cuda(batch['input'], cuda_available=cuda_available)

            if ('FiLMedUnet' in context and context['FiLMedUnet']['applied']) or \
                    ('HeMISUnet' in context and context['HeMISUnet']['applied']):
                metadata = imed_training.get_metadata(batch["input_metadata"], model_params)
                preds = model(img, metadata)

            else:
                preds = model(img) if fname_model.endswith('.pt') else onnx_inference(fname_model, img)

            preds = preds.cpu()

        # Set datatype to gt since prediction should be processed the same way as gt
        for b in batch['input_metadata']:
            for modality in b:
                modality['data_type'] = 'gt'

        # Reconstruct 3D object
        for i_slice in range(len(preds)):
            if "bounding_box" in batch['input_metadata'][i_slice][0]:
                imed_obj_detect.adjust_undo_transforms(undo_transforms.transforms, batch, i_slice)

            batch['gt_metadata'] = [[metadata[0]] * preds.shape[1] for metadata in batch['input_metadata']]
            if kernel_3D:
                preds_undo, metadata, last_sample_bool, volume, weight_matrix = \
                    volume_reconstruction(batch, preds, undo_transforms, i_slice, volume, weight_matrix)
                preds_list = [np.array(preds_undo)]
            else:
                # undo transformations
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
        class_pred = nib.Nifti1Image(pred[..., c].astype('float32'), nib_prediction.affine)
        pred_list.append(class_pred)
    return pred_list


def volume_reconstruction(batch, pred, undo_transforms, smp_idx, volume=None, weight_matrix=None):
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
        tensor, dict, bool, tensor, tensor: undone subvolume, metadata, boolean representing if its the last sample to
        process, reconstructed volume, weight matrix
    """
    x_min, x_max, y_min, y_max, z_min, z_max = batch['input_metadata'][smp_idx][0]['coord']
    num_pred = pred[smp_idx].shape[0]

    first_sample_bool = not any([x_min, y_min, z_min])
    x, y, z = batch['input_metadata'][smp_idx][0]['index_shape']
    if first_sample_bool:
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


def onnx_inference(model_path, inputs):
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

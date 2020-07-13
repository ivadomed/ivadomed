import json
import os

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from ivadomed import models as imed_models
from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader
from ivadomed.object_detection import utils as imed_obj_detect
from scipy.ndimage import label, generate_binary_structure
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

AXIS_DCT = {'sagittal': 0, 'coronal': 1, 'axial': 2}


def pred_to_nib(data_lst, z_lst, fname_ref, fname_out, slice_axis, debug=False, kernel_dim='2d', bin_thr=0.5,
                discard_noise=True):
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
            oriented_volumes.append(reorient_image(arr_pred_ref_space[i,], slice_axis, nib_ref, nib_ref_can))
        # transpose to locate the channel dimension at the end to properly see image on viewer
        arr_pred_ref_space = np.asarray(oriented_volumes).transpose((1, 2, 3, 0))
    else:
        arr_pred_ref_space = reorient_image(arr_pred_ref_space, slice_axis, nib_ref, nib_ref_can)

    if bin_thr >= 0:
        arr_pred_ref_space = imed_postpro.threshold_predictions(arr_pred_ref_space, thr=bin_thr)
    elif discard_noise:  # discard noise
        arr_pred_ref_space[arr_pred_ref_space <= 1e-2] = 0

    # create nibabel object
    nib_pred = nib.Nifti1Image(arr_pred_ref_space, nib_ref.affine)

    # save as nifti file
    if fname_out is not None:
        nib.save(nib_pred, fname_out)

    return nib_pred


def run_uncertainty(ifolder):
    """Compute uncertainty from model prediction.

    This function loops across the model predictions (nifti masks) and estimates the uncertainty from the Monte Carlo
    samples. Both voxel-wise and structure-wise uncertainty are estimates.

    Args:
        ifolder (str): Folder containing the Monte Carlo samples.
    """
    # list subj_acq prefixes
    subj_acq_lst = [f.split('_pred')[0] for f in os.listdir(ifolder)
                    if f.endswith('.nii.gz') and '_pred' in f]
    # remove duplicates
    subj_acq_lst = list(set(subj_acq_lst))
    # keep only the images where unc has not been computed yet
    subj_acq_lst = [f for f in subj_acq_lst if not os.path.isfile(
        os.path.join(ifolder, f + '_unc-cv.nii.gz'))]

    # loop across subj_acq
    for subj_acq in tqdm(subj_acq_lst, desc="Uncertainty Computation"):
        # hard segmentation from MC samples
        fname_pred = os.path.join(ifolder, subj_acq + '_pred.nii.gz')
        # fname for soft segmentation from MC simulations
        fname_soft = os.path.join(ifolder, subj_acq + '_soft.nii.gz')
        # find Monte Carlo simulations
        fname_pred_lst = [os.path.join(ifolder, f)
                          for f in os.listdir(ifolder) if subj_acq + '_pred_' in f]

        # if final segmentation from Monte Carlo simulations has not been generated yet
        if not os.path.isfile(fname_pred) or not os.path.isfile(fname_soft):
            # threshold used for the hard segmentation
            thr = 1. / len(fname_pred_lst)  # 1 for all voxels where at least on MC sample predicted 1
            # average then argmax
            combine_predictions(fname_pred_lst, fname_pred, fname_soft, thr=thr)

        fname_unc_vox = os.path.join(ifolder, subj_acq + '_unc-vox.nii.gz')
        if not os.path.isfile(fname_unc_vox):
            # compute voxel-wise uncertainty map
            voxelwise_uncertainty(fname_pred_lst, fname_unc_vox)

        fname_unc_struct = os.path.join(ifolder, subj_acq + '_unc.nii.gz')
        if not os.path.isfile(os.path.join(ifolder, subj_acq + '_unc-cv.nii.gz')):
            # compute structure-wise uncertainty
            structurewise_uncertainty(fname_pred_lst, fname_pred, fname_unc_vox, fname_unc_struct)


def combine_predictions(fname_lst, fname_hard, fname_prob, thr=0.5):
    """Combine predictions from Monte Carlo simulations.

    Combine predictions from Monte Carlo simulations and save the resulting as:
        (1) `fname_prob`, a soft segmentation obtained by averaging the Monte Carlo samples.
        (2) `fname_hard`, a hard segmentation obtained thresholding with `thr`.

    Args:
        fname_lst (list of str): List of the Monte Carlo samples.
        fname_hard (str): Filename for the output hard segmentation.
        fname_prob (str): Filename for the output soft segmentation.
        thr (float): Between 0 and 1. Used to threshold the soft segmentation and generate the hard segmentation.
    """
    # collect all MC simulations
    data_lst = []
    for fname in fname_lst:
        nib_im = nib.load(fname)
        data_lst.append(nib_im.get_fdata())

    # average over all the MC simulations
    data_prob = np.mean(np.array(data_lst), axis=0)
    # save prob segmentation
    nib_prob = nib.Nifti1Image(data_prob, nib_im.affine)
    nib.save(nib_prob, fname_prob)

    # argmax operator
    # TODO: adapt for multi-label pred
    data_hard = imed_postpro.threshold_predictions(data_prob, thr=thr).astype(np.uint8)
    # save hard segmentation
    nib_hard = nib.Nifti1Image(data_hard, nib_im.affine)
    nib.save(nib_hard, fname_hard)


def voxelwise_uncertainty(fname_lst, fname_out, eps=1e-5):
    """Estimate voxel wise uncertainty.

    Voxel-wise uncertainty is estimated as entropy over all N MC probability maps, and saved in `fname_out`.

    Args:
        fname_lst (list of str): List of the Monte Carlo samples.
        fname_out (str): Output filename.
        eps (float): Epsilon value to deal with np.log(0).
    """
    # collect all MC simulations
    data_lst = []
    for fname in fname_lst:
        nib_im = nib.load(fname)
        data_lst.append(nib_im.get_fdata())

    # entropy
    unc = np.repeat(np.expand_dims(np.array(data_lst), -1), 2, -1)  # n_it, x, y, z, 2
    unc[..., 0] = 1 - unc[..., 1]
    unc = -np.sum(np.mean(unc, 0) * np.log(np.mean(unc, 0) + eps), -1)

    # save uncertainty map
    nib_unc = nib.Nifti1Image(unc, nib_im.affine)
    nib.save(nib_unc, fname_out)


def structurewise_uncertainty(fname_lst, fname_hard, fname_unc_vox, fname_out):
    """Estimate structure wise uncertainty.

    Structure-wise uncertainty from N MC probability maps (`fname_lst`) and saved in `fname_out` with the following
    suffixes:

        * '-cv.nii.gz': coefficient of variation
        * '-iou.nii.gz': intersection over union
        * '-avgUnc.nii.gz': average voxel-wise uncertainty within the structure.

    Args:
        fname_lst (list of str): List of the Monte Carlo samples.
        fname_hard (str): Filename of the hard segmentation, which is used to compute the `avgUnc` by providing a mask
            of the structures.
        fname_unc_vox (str): Filename of the voxel-wise uncertainty, which is used to compute the `avgUnc`.
        fname_out (str): Output filename.
    """
    # load hard segmentation and label it
    nib_hard = nib.load(fname_hard)
    data_hard = nib_hard.get_fdata()
    bin_struct = generate_binary_structure(3, 2)  # 18-connectivity
    data_hard_l, n_l = label(data_hard, structure=bin_struct)

    # load uncertainty map
    nib_uncVox = nib.load(fname_unc_vox)
    data_uncVox = nib_uncVox.get_fdata()
    del nib_uncVox

    # init output arrays
    data_iou, data_cv, data_avgUnc = np.zeros(data_hard.shape), np.zeros(data_hard.shape), np.zeros(data_hard.shape)

    # load all MC simulations and label them
    data_lst, data_l_lst = [], []
    for fname in fname_lst:
        nib_im = nib.load(fname)
        data_im = nib_im.get_fdata()
        data_lst.append(data_im)
        data_im_l, _ = label(data_im, structure=bin_struct)
        data_l_lst.append(data_im_l)
        del nib_im

    # loop across all structures of data_hard_l
    for i_l in range(1, n_l + 1):
        # select the current structure, remaining voxels are set to zero
        data_i_l = (data_hard_l == i_l).astype(np.int)

        # select the current structure in each MC sample
        # and store it in data_mc_i_l_lst
        data_mc_i_l_lst = []
        # loop across MC samples
        for i_mc in range(len(data_lst)):
            # find the structure of interest in the current MC sample
            data_i_inter = data_i_l * data_l_lst[i_mc]
            i_mc_l = np.max(data_i_inter)

            if i_mc_l > 0:
                # keep only the unc voxels of the structure of interest
                data_mc_i_l = np.copy(data_lst[i_mc])
                data_mc_i_l[data_l_lst[i_mc] != i_mc_l] = 0.
            else:  # no structure in this sample
                data_mc_i_l = np.zeros(data_lst[i_mc].shape)
            data_mc_i_l_lst.append(data_mc_i_l)

        # compute IoU over all the N MC samples for a specific structure
        intersection = np.logical_and(data_mc_i_l_lst[0].astype(np.bool),
                                      data_mc_i_l_lst[1].astype(np.bool))
        union = np.logical_or(data_mc_i_l_lst[0].astype(np.bool),
                              data_mc_i_l_lst[1].astype(np.bool))
        for i_mc in range(2, len(data_mc_i_l_lst)):
            intersection = np.logical_and(intersection,
                                          data_mc_i_l_lst[i_mc].astype(np.bool))
            union = np.logical_or(union,
                                  data_mc_i_l_lst[i_mc].astype(np.bool))
        iou = np.sum(intersection) * 1. / np.sum(union)

        # compute coefficient of variation for all MC volume estimates for a given structure
        vol_mc_lst = [np.sum(data_mc_i_l_lst[i_mc]) for i_mc in range(len(data_mc_i_l_lst))]
        mu_mc = np.mean(vol_mc_lst)
        sigma_mc = np.std(vol_mc_lst)
        cv = sigma_mc / mu_mc

        # compute average voxel-wise uncertainty within the structure
        avgUnc = np.mean(data_uncVox[data_i_l != 0])
        # assign uncertainty value to the structure
        data_iou[data_i_l != 0] = iou
        data_cv[data_i_l != 0] = cv
        data_avgUnc[data_i_l != 0] = avgUnc

    # save nifti files
    fname_iou = fname_out.split('.nii.gz')[0] + '-iou.nii.gz'
    fname_cv = fname_out.split('.nii.gz')[0] + '-cv.nii.gz'
    fname_avgUnc = fname_out.split('.nii.gz')[0] + '-avgUnc.nii.gz'
    nib_iou = nib.Nifti1Image(data_iou, nib_hard.affine)
    nib_cv = nib.Nifti1Image(data_cv, nib_hard.affine)
    nib_avgUnc = nib.Nifti1Image(data_avgUnc, nib_hard.affine)
    nib.save(nib_iou, fname_iou)
    nib.save(nib_cv, fname_cv)
    nib.save(nib_avgUnc, fname_avgUnc)


def mixup(data, targets, alpha, debugging=False, ofolder=None):
    """Compute the mixup data.

    .. seealso::
        Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization."
        arXiv preprint arXiv:1710.09412 (2017).

    Args:
        data (Tensor): Input images.
        targets (Tensor): Input masks.
        alpha (float): MixUp parameter.
        debugging (Bool): If True, then samples of mixup are saved as png files.
        ofolder (str): If debugging, output folder where "mixup" folder is created and samples are saved.

    Returns:
        Tensor, Tensor: Mixed image, Mixed mask.
    """
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lambda_ = np.random.beta(alpha, alpha)
    lambda_ = max(lambda_, 1 - lambda_)  # ensure lambda_ >= 0.5
    lambda_tensor = torch.FloatTensor([lambda_])


    data = data * lambda_tensor + data2 * (1 - lambda_tensor)
    targets = targets * lambda_tensor + targets2 * (1 - lambda_tensor)

    if debugging:
        save_mixup_sample(ofolder, data, targets, lambda_tensor)

    return data, targets


def save_mixup_sample(ofolder, input_data, labeled_data, lambda_tensor):
    """Save mixup samples as png files in a "mixup" folder.

    Args:
        ofolder (str): Output folder where "mixup" folder is created and samples are saved.
        input_data (Tensor): Input image.
        labeled_data (Tensor): Input masks.
        lambda_tensor (Tensor):
    """
    # Mixup folder
    mixup_folder = os.path.join(ofolder, 'mixup')
    if not os.path.isdir(mixup_folder):
        os.makedirs(mixup_folder)
    # Random sample
    random_idx = np.random.randint(0, input_data.size()[0])
    # Output fname
    ofname = str(lambda_tensor.data.numpy()[0]) + '_' + str(random_idx).zfill(3) + '.png'
    ofname = os.path.join(mixup_folder, ofname)
    # Tensor to Numpy
    x = input_data.data.numpy()[random_idx, 0, :, :]
    y = labeled_data.data.numpy()[random_idx, 0, :, :]
    # Plot
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(x, interpolation='nearest', aspect='auto', cmap='gray')
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(y, interpolation='nearest', aspect='auto', cmap='jet', vmin=0, vmax=1)
    plt.savefig(ofname, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


def segment_volume(folder_model, fname_image, fname_prior=None, gpu_number=0):
    """Segment an image.

    Segment an image (`fname_image`) using a pre-trained model (`folder_model`). If provided, a region of interest
    (`fname_roi`) is used to crop the image prior to segment it.

    Args:
        folder_model (str): foldername which contains
            (1) the model ('folder_model/folder_model.pt') to use
            (2) its configuration file ('folder_model/folder_model.json') used for the training,
            see https://github.com/neuropoly/ivadomed/wiki/configuration-file
        fname_image (str): image filename (e.g. .nii.gz) to segment.
        fname_prior (str): Image filename (e.g. .nii.gz) containing processing information (e.i. spinal cord
            segmentation, spinal location or MS lesion classification)

            e.g. spinal cord centerline, used to crop the image prior to segment it if provided.
            The segmentation is not performed on the slices that are empty in this image.
        gpu_number (int): Number representing gpu number if available.

    Returns:
        nibabelObject: Object containing the soft segmentation.
    """
    # Define device
    cuda_available = torch.cuda.is_available()
    device = torch.device("cpu") if not cuda_available else torch.device("cuda:" + str(gpu_number))

    # Check if model folder exists and get filenames
    fname_model, fname_model_metadata = imed_models.get_model_filenames(folder_model)

    # Load model training config
    with open(fname_model_metadata, "r") as fhandle:
        context = json.load(fhandle)

    # LOADER
    loader_params = context["loader_parameters"]
    slice_axis = AXIS_DCT[loader_params['slice_axis']]
    metadata = {}
    fname_roi = None
    if fname_prior is not None:
        if 'roi_params' in loader_params and loader_params['roi_params']['suffix'] is not None:
            fname_roi = fname_prior
        # TRANSFORMATIONS
        # If ROI is not provided then force center cropping
        if fname_roi is None and 'ROICrop' in context["transformation"].keys():
            print(
                "\nWARNING: fname_roi has not been specified, then a cropping around the center of the image is performed"
                " instead of a cropping around a Region of Interest.")
            context["transformation"] = dict((key, value) if key != 'ROICrop'
                                             else ('CenterCrop', value)
                                             for (key, value) in context["transformation"].items())

        if 'object_detection_params' in context and \
                context['object_detection_params']['object_detection_path'] is not None:
            imed_obj_detect.bounding_box_prior(fname_prior, metadata, slice_axis)

    # Compose transforms
    _, _, transform_test_params = imed_transforms.get_subdatasets_transforms(context["transformation"])

    tranform_lst, undo_transforms = imed_transforms.prepare_transforms(transform_test_params)

    # Force filter_empty_mask to False if fname_roi = None
    if fname_roi is None and 'filter_empty_mask' in loader_params["slice_filter_params"]:
        print("\nWARNING: fname_roi has not been specified, then the entire volume is processed.")
        loader_params["slice_filter_params"]["filter_empty_mask"] = False

    filename_pairs = [([fname_image], None, fname_roi, [metadata])]

    kernel_3D = bool('UNet3D' in context and context['UNet3D']['applied'])
    if kernel_3D:
        ds = imed_loader.MRI3DSubVolumeSegmentationDataset(filename_pairs,
                                                           transform=tranform_lst,
                                                           length=context["UNet3D"]["length_3D"],
                                                           stride=context["UNet3D"]["stride_3D"])
    else:
        ds = imed_loader.MRI2DSegmentationDataset(filename_pairs,
                                                  slice_axis=slice_axis,
                                                  cache=True,
                                                  transform=tranform_lst,
                                                  slice_filter_fn=SliceFilter(**loader_params["slice_filter_params"]))
        ds.load_filenames()

    if kernel_3D:
        print("\nLoaded {} {} volumes of shape {}.".format(len(ds), loader_params['slice_axis'],
                                                           context['UNet3D']['length_3D']))
    else:
        print("\nLoaded {} {} slices.".format(len(ds), loader_params['slice_axis']))

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
            img = cuda(batch['input'], cuda_available=cuda_available)
            preds = model(img) if fname_model.endswith('.pt') else onnx_inference(fname_model, img)
            preds = preds.cpu()

        # Set datatype to gt since prediction should be processed the same way as gt
        for modality in batch['input_metadata']:
            modality[0]['data_type'] = 'gt'

        # Reconstruct 3D object
        for i_slice in range(len(preds)):
            if "bounding_box" in batch['input_metadata'][i_slice][0]:
                imed_obj_detect.adjust_undo_transforms(undo_transforms.transforms, batch, i_slice)

            if kernel_3D:
                batch['gt_metadata'] = batch['input_metadata']
                preds_undo, metadata, last_sample_bool, volume, weight_matrix = \
                    volume_reconstruction(batch, preds, undo_transforms, i_slice, volume, weight_matrix)
                preds_list = [np.array(preds_undo)]
            else:
                # undo transformations
                preds_i_undo, metadata_idx = undo_transforms(preds[i_slice],
                                                             batch["input_metadata"][i_slice],
                                                             data_type='gt')

                # Add new segmented slice to preds_list
                preds_list.append(np.array(preds_i_undo))
                # Store the slice index of preds_i_undo in the original 3D image
                slice_idx_list.append(int(batch['input_metadata'][i_slice][0]['slice_index']))

            # If last batch and last sample of this batch, then reconstruct 3D object
            if (i_batch == len(data_loader) - 1 and i_slice == len(batch['gt']) - 1) or last_sample_bool:
                pred_nib = pred_to_nib(data_lst=preds_list,
                                       fname_ref=fname_image,
                                       fname_out=None,
                                       z_lst=slice_idx_list,
                                       slice_axis=slice_axis,
                                       kernel_dim='3d' if kernel_3D else '2d',
                                       debug=False,
                                       bin_thr=-1)

    return pred_nib


def cuda(input_var, cuda_available=True, non_blocking=False):
    """Passes input_var to GPU.

    Args:
        input_var (Tensor): either a tensor or a list of tensors.
        cuda_available (bool): If False, then return identity
        non_blocking (bool):

    Returns:
        Tensor
    """
    if cuda_available:
        if isinstance(input_var, list):
            return [t.cuda(non_blocking=non_blocking) for t in input_var]
        else:
            return input_var.cuda(non_blocking=non_blocking)
    else:
        return input_var


class HookBasedFeatureExtractor(nn.Module):
    """This function extracts feature maps from given layer. Helpful to observe where the attention of the network is
    focused.

    https://github.com/ozan-oktay/Attention-Gated-Networks/tree/a96edb72622274f6705097d70cfaa7f2bf818a5a

    Args:
        submodule (nn.Module): Trained model.
        layername (str): Name of the layer where features need to be extracted (layer of interest).
        upscale (bool): If True output is rescaled to initial size.

    Attributes:
        submodule (nn.Module): Trained model.
        layername (str):  Name of the layer where features need to be extracted (layer of interest).
        outputs_size (list): List of output sizes.
        outputs (list): List of outputs containing the features of the given layer.
        inputs (list): List of inputs.
        inputs_size (list): List of input sizes.
    """

    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, i):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


def reorient_image(arr, slice_axis, nib_ref, nib_ref_canonical):
    """Reorient an image to match a reference image orientation.

    It reorients a array to a given orientation and convert it to a nibabel object using the reference nibabel header.

    Args:
        arr (ndarray): Input array, array to re orient.
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
        nib_ref (nibabel): Reference nibabel object, whose header is used.
        nib_ref_canonical (nibabel): `nib_ref` that has been reoriented to canonical orientation (RAS).
    """
    # Orient image in RAS according to slice axis
    arr_ras = imed_loader_utils.orient_img_ras(arr, slice_axis)

    # https://gitship.com/neuroscience/nibabel/blob/master/nibabel/orientations.py
    ref_orientation = nib.orientations.io_orientation(nib_ref.affine)
    ras_orientation = nib.orientations.io_orientation(nib_ref_canonical.affine)
    # Return the orientation that transforms from ras to ref_orientation
    trans_orient = nib.orientations.ornt_transform(ras_orientation, ref_orientation)
    # apply transformation
    return nib.orientations.apply_orientation(arr_ras, trans_orient)


def save_feature_map(batch, layer_name, log_directory, model, test_input, slice_axis):
    """Save model feature maps.

    Args:
        batch (dict):
        layer_name (str):
        log_directory (str): Output folder.
        model (nn.Module): Network.
        test_input (Tensor):
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
    """
    if not os.path.exists(os.path.join(log_directory, layer_name)):
        os.mkdir(os.path.join(log_directory, layer_name))

    # Save for subject in batch
    for i in range(batch['input'].size(0)):
        inp_fmap, out_fmap = \
            HookBasedFeatureExtractor(model, layer_name, False).forward(Variable(test_input[i][None,]))

        # Display the input image and Down_sample the input image
        orig_input_img = test_input[i][None,].cpu().numpy()
        upsampled_attention = F.interpolate(out_fmap[1],
                                            size=test_input[i][None,].size()[2:],
                                            mode='trilinear',
                                            align_corners=True).data.cpu().numpy()

        path = batch["input_metadata"][0][i]["input_filenames"]

        basename = path.split('/')[-1]
        save_directory = os.path.join(log_directory, layer_name, basename)

        # Write the attentions to a nifti image
        nib_ref = nib.load(path)
        nib_ref_can = nib.as_closest_canonical(nib_ref)
        oriented_image = reorient_image(orig_input_img[0, 0, :, :, :], slice_axis, nib_ref, nib_ref_can)

        nib_pred = nib.Nifti1Image(oriented_image, nib_ref.affine)
        nib.save(nib_pred, save_directory)

        basename = basename.split(".")[0] + "_att.nii.gz"
        save_directory = os.path.join(log_directory, layer_name, basename)
        attention_map = reorient_image(upsampled_attention[0, 0, :, :, :], slice_axis, nib_ref, nib_ref_can)
        nib_pred = nib.Nifti1Image(attention_map, nib_ref.affine)

        nib.save(nib_pred, save_directory)


def save_color_labels(gt_data, binarize, gt_filename, output_filename, slice_axis):
    """Saves labels encoded in RGB in specified output file.

    Args:
        gt_data (ndarray): Input image with dimensions (Number of classes, height, width, depth).
        binarize (bool): If True binarizes gt_data to 0 and 1 values, else soft values are kept.
        gt_filename (str): GT path and filename.
        output_filename (str): Name of the output file where the colored labels are saved.
        slice_axis (int): Indicates the axis used to extract slices: "axial": 2, "sagittal": 0, "coronal": 1.

    Returns:
        ndarray: RGB labels.
    """
    n_class, h, w, d = gt_data.shape
    labels = range(n_class)
    # Generate color labels
    multi_labeled_pred = np.zeros((h, w, d, 3))
    if binarize:
        gt_data = imed_postpro.threshold_predictions(gt_data)

    # Keep always the same color labels
    np.random.seed(6)

    for label in labels:
        r, g, b = np.random.randint(0, 256, size=3)
        multi_labeled_pred[..., 0] += r * gt_data[label,]
        multi_labeled_pred[..., 1] += g * gt_data[label,]
        multi_labeled_pred[..., 2] += b * gt_data[label,]

    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    multi_labeled_pred = multi_labeled_pred.copy().astype('u1').view(dtype=rgb_dtype).reshape((h, w, d))

    pred_to_nib([multi_labeled_pred], [], gt_filename,
                output_filename, slice_axis=slice_axis, kernel_dim='3d', bin_thr=-1, discard_noise=False)

    return multi_labeled_pred


def convert_labels_to_RGB(grid_img):
    """Converts 2D images to RGB encoded images for display in tensorboard.

    Args:
        grid_img (Tensor): GT or prediction tensor with dimensions (batch size, number of classes, height, width).

    Returns:
        tensor: RGB image with shape (height, width, 3).
    """
    # Keep always the same color labels
    batch_size, n_class, h, w = grid_img.shape
    rgb_img = torch.zeros((batch_size, 3, h, w))

    # Keep always the same color labels
    np.random.seed(6)
    for i in range(n_class):
        r, g, b = np.random.randint(0, 256, size=3)
        rgb_img[:, i, ] = r * grid_img[:, i, ]
        rgb_img[:, i, ] = g * grid_img[:, i, ]
        rgb_img[:, i, ] = b * grid_img[:, i, ]

    return rgb_img


def save_tensorboard_img(writer, epoch, dataset_type, input_samples, gt_samples, preds, is_three_dim=False):
    """Saves input images, gt and predictions in tensorboard.

    Args:
        writer (SummaryWriter): Tensorboard's summary writer.
        epoch (int): Epoch number.
        dataset_type (str): Choice between Training or Validation.
        input_samples (Tensor): Input images with shape (batch size, number of channel, height, width, depth) if 3D else
            (batch size, number of channel, height, width)
        gt_samples (Tensor): GT images with shape (batch size, number of channel, height, width, depth) if 3D else
            (batch size, number of channel, height, width)
        preds (Tensor): Model's prediction with shape (batch size, number of channel, height, width, depth) if 3D else
            (batch size, number of channel, height, width)
        is_three_dim (bool): True if 3D input, else False.
    """
    if is_three_dim:
        # Take all images stacked on depth dimension
        num_2d_img = input_samples.shape[-1]
    else:
        num_2d_img = 1
    if isinstance(input_samples, list):
        input_samples_copy = input_samples.copy()
    else:
        input_samples_copy = input_samples.clone()
    preds_copy = preds.clone()
    gt_samples_copy = gt_samples.clone()
    for idx in range(num_2d_img):
        if is_three_dim:
            input_samples = input_samples_copy[..., idx]
            preds = preds_copy[..., idx]
            gt_samples = gt_samples_copy[..., idx]
            # Only display images with labels
            if gt_samples.sum() == 0:
                continue

        # take only one modality for grid
        if not isinstance(input_samples, list) and input_samples.shape[1] > 1:
            tensor = input_samples[:, 0, ][:, None, ]
            input_samples = torch.cat((tensor, tensor, tensor), 1)
        elif isinstance(input_samples, list):
            input_samples = input_samples[0]

        grid_img = vutils.make_grid(input_samples,
                                    normalize=True,
                                    scale_each=True)
        writer.add_image(dataset_type + '/Input', grid_img, epoch)

        grid_img = vutils.make_grid(convert_labels_to_RGB(preds),
                                    normalize=True,
                                    scale_each=True)

        writer.add_image(dataset_type + '/Predictions', grid_img, epoch)

        grid_img = vutils.make_grid(convert_labels_to_RGB(gt_samples),
                                    normalize=True,
                                    scale_each=True)

        writer.add_image(dataset_type + '/Ground Truth', grid_img, epoch)


class SliceFilter(object):
    """Filter 2D slices from dataset.

    If a sample does not meet certain conditions, it is discarded from the dataset.

    Args:
        filter_empty_mask (bool): If True, samples where all voxel labels are zeros are discarded.
        filter_empty_input (bool): If True, samples where all voxel intensities are zeros are discarded.

    Attributes:
        filter_empty_mask (bool): If True, samples where all voxel labels are zeros are discarded.
        filter_empty_input (bool): If True, samples where all voxel intensities are zeros are discarded.
    """
    def __init__(self, filter_empty_mask=True,
                 filter_empty_input=True):
        self.filter_empty_mask = filter_empty_mask
        self.filter_empty_input = filter_empty_input

    def __call__(self, sample):
        input_data, gt_data = sample['input'], sample['gt']

        if self.filter_empty_mask:
            if not np.any(gt_data):
                return False

        if self.filter_empty_input:
            if not np.all([np.any(img) for img in input_data]):
                return False

        return True


def unstack_tensors(sample):
    """Unstack tensors.

    Args:
        sample (Tensor):

    Returns:
        list: list of Tensors.
    """
    list_tensor = []
    for i in range(sample.shape[1]):
        list_tensor.append(sample[:, i, ].unsqueeze(1))
    return list_tensor


def save_onnx_model(model, inputs, model_path):
    """Convert PyTorch model to ONNX model and save it as `model_path`.

    Args:
        model (nn.Module): PyTorch model.
        inputs (Tensor): Tensor, used to inform shape and axes.
        model_path (str): Output filename for the ONNX model.
    """
    model.eval()
    dynamic_axes = {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width', 4: 'depth'}
    if len(inputs.shape) == 4:
        del dynamic_axes[4]
    torch.onnx.export(model, inputs, model_path,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': dynamic_axes, 'output': dynamic_axes})


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


def define_device(gpu_id):
    """Define the device used for the process of interest.

    Args:
        gpu_id (int): GPU ID.

    Returns:
        Bool, device: True if cuda is available.
    """
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("Cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_number = int(gpu_id)
        torch.cuda.set_device(gpu_number)
        print("Using GPU number {}".format(gpu_number))
    return cuda_available, device


def display_selected_model_spec(params):
    """Display in terminal the selected model and its parameters.

    Args:
        params (dict): Keys are param names and values are param values.
    """
    print('\nSelected architecture: {}, with the following parameters:'.format(params["name"]))
    for k in list(params.keys()):
        if k != "name":
            print('\t{}: {}'.format(k, params[k]))


def display_selected_transfoms(params, dataset_type):
    """Display in terminal the selected transforms for a given dataset.

    Args:
        params (dict):
        dataset_type (list): e.g. ['testing'] or ['training', 'validation']
    """
    print('\nSelected transformations for the {} dataset:'.format(dataset_type))
    for k in list(params.keys()):
        print('\t{}: {}'.format(k, params[k]))


def plot_transformed_sample(before, after, list_title=[], fname_out="", cmap="jet"):
    """Utils tool to plot sample before and after transform, for debugging.

    Args:
        before (ndarray): Sample before transform.
        after (ndarray): Sample after transform.
        list_title (list of str): Sub titles of before and after, resp.
        fname_out (str): Output filename where the plot is saved if provided.
        cmap (str): Matplotlib colour map.
    """
    if len(list_title) == 0:
        list_title = ['Sample before transform', 'Sample after transform']

    plt.interactive(False)
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(before, interpolation='nearest', cmap=cmap)
    plt.title(list_title[0], fontsize=20)

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(after, interpolation='nearest', cmap=cmap)
    plt.title(list_title[1], fontsize=20)

    if fname_out:
        plt.savefig(fname_out)
    else:
        matplotlib.use('TkAgg')
        plt.show()


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

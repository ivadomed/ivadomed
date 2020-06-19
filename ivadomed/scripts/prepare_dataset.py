import argparse
import ivadomed.utils as imed_utils
import ivadomed.preprocessing as imed_preprocessing
import nibabel as nib
import numpy as np
import os
import scipy
import skimage


# normalize between 0 and 1.
def normalize(arr):
    ma = arr.max()
    mi = arr.min()
    return (arr - mi) / (ma - mi)


def gaussian_kernel(kernlen=10):
    """
    Create a 2D gaussian kernel with user-defined size.

    Args:
        kernlen(int): size of kernel

    Returns:
        array: a 2D array of size (kernlen,kernlen)
    """

    x = np.linspace(-1, 1, kernlen + 1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return normalize(kern2d / kern2d.sum())


def heatmap_generation(image, kernel_size):
    """
    Generate heatmap from image containing sing voxel label using
    convolution with gaussian kernel
    Args:
        image: 2D array containing single voxel label
        kernel_size: size of gaussian kernel

    Returns:
        array: 2D array heatmap matching the label.

    """
    kernel = gaussian_kernel(kernel_size)
    map = scipy.signal.convolve(image, kernel, mode='same')
    return normalize(map)


def mask2label(path_label, aim='full'):
    """
    Retrieve points coordinates and value from a label file containing singl voxel label
    Args:
        path_label: path of nifti image
        aim: 'full' or 'c2' full will return all points with label between 3 and 30 , c2 will return only the coordinates of points label 3

    Returns:
        array: array containing the asked point in the format [x,y,z,value]

    """
    image = nib.load(path_label)
    image = nib.as_closest_canonical(image)
    arr = np.array(image.dataobj)
    list_label_image = []
    # Arr non zero used since these are single voxel label
    for i in range(len(arr.nonzero()[0])):
        x = arr.nonzero()[0][i]
        y = arr.nonzero()[1][i]
        z = arr.nonzero()[2][i]
        # need to check every points
        if aim == 'full':
            # we don't want to account for pmj (label 49) nor C1/C2 which is hard to distinguish.
            if arr[x, y, z] < 30 and arr[x, y, z] != 1:
                list_label_image.append([x, y, z, arr[x, y, z]])
        elif aim == 'c2':
            if arr[x, y, z] == 3:
                list_label_image.append([x, y, z, arr[x, y, z]])
    list_label_image.sort(key=lambda x: x[3])
    return list_label_image


def extract_mid_slice_and_convert_coordinates_to_heatmaps(bids_path, suffix, aim="full"):
    """
     This function takes as input a path to a dataset  and generates two sets of images:
   (i) mid-sagittal image of common size (1,ap_pad,is_pad) and
   (ii) heatmap of disc labels associated with the mid-sagittal image.

    Args:
        bids_path (string): path to BIDS dataset form which images will be generated
        suffix (string): suffix of image that will be processed (e.g., T2w)
        aim(string): 'full' or 'c2'. If 'c2' retrieves only c2 label (value = 3) else create heatmap with all label.
    Returns:
        None. Images are saved in BIDS folder
    """
    t = os.listdir(bids_path)
    t.remove('derivatives')

    for i in range(len(t)):
        sub = t[i]
        path_image = bids_path + t[i] + '/anat/' + t[i] + suffix + '.nii.gz'
        if os.path.isfile(path_image):
            path_label = bids_path + 'derivatives/labels/' + t[i] + '/anat/' + t[i] + suffix \
                         + '_label-disc-manual.nii.gz'
            list_points = mask2label(path_label, aim=aim)
            image_ref = nib.load(path_image)
            nib_ref_can = nib.as_closest_canonical(image_ref)
            imsh = np.array(nib_ref_can.dataobj).shape
            mid_nifti = imed_preprocessing.get_midslice_average(path_image, list_points[0][0], slice_axis=0)
            nib.save(mid_nifti, bids_path + t[i] + '/anat/' + t[i] + suffix + '_mid.nii.gz')
            lab = nib.load(path_label)
            nib_ref_can = nib.as_closest_canonical(lab)
            label_array = np.zeros(imsh[1:])

            for j in range(len(list_points)):
                label_array[list_points[j][1], list_points[j][2]] = 1

            heatmap = heatmap_generation(label_array[:, :], 10)
            nib_pred = nib.Nifti1Image(heatmap, lab.affine)
            nib.save(nib_pred,
                     bids_path + 'derivatives/labels/' + t[i] + '/anat/' + t[i] + suffix + '_mid_heatmap.nii.gz')
        else:
            pass


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", dest="path", required=True, type=str, help="Path to bids folder with "/" at the end.")
    parser.add_argument("-s", "--suffix", dest="suffix", required=True,
                        type=str, help="suffix of image file")
    parser.add_argument("-a", "--aim", dest="aim", default="full", type=str, help="If set to 'c2' only points with value 3 will be converted to heatmap")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    bids_path = args.path
    suffix = args.suffix
    aim = args.aim
    # Run Script
    extract_mid_slice_and_convert_coordinates_to_heatmaps(bids_path, suffix, aim)

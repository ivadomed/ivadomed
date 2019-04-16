
import argparse
import numpy as np
import nibabel as nib


def remove_slice(fname_im, fname_seg):
    """
    Look at the average signal within the segmentation, for edge slices, and remove those slices if this average is
    zero. Assumes last dimension is Z.
    :param fname_im:
    :param fname_seg:
    :return:
    """
    # Load data
    nii_im = nib.load(fname_im)
    data_im = nii_im.get_data()
    affine_im = nii_im.affine
    nii_seg = nib.load(fname_seg)
    data_seg = nii_seg.get_data()
    _, _, nz = nii_im.shape
    # Loop across slices in ascending mode and stop when no more empty
    z_bottom = 0
    for iz in range(nz):
        if np.mean(data_im[:, :, iz] * data_seg[:, :, iz]) == 0.0:
            z_bottom = iz + 1
        else:
            break
    # Loop across slices in descending mode and stop when no more empty
    z_top = 0
    for iz in range(nz-1, -1, -1):
        if np.mean(data_im[:, :, iz] * data_seg[:, :, iz]) == 0.0:
            z_top = iz
        else:
            break
    # If some slices are empty, crop image and segmentation
    if z_bottom or z_top:
        # Remove edge slices
        data_im_crop = data_im[..., z_bottom:z_top]
        data_seg_crop = data_seg[..., z_bottom:z_top]
        # Calculate the translation (in voxel space)
        translation_vox = [0, 0, z_bottom]
        # Update affine transformation, accounting for the number of slices removed (voxel coordinate system is shifted)
        transfo = affine_im[0:3, 0:3]
        translation = np.dot(transfo, np.transpose(translation_vox))
        affine_im_new = np.copy(affine_im)
        affine_im_new[0:3, 3] = affine_im_new[0:3, 3] + translation
        # Overwrite image and segmentation
        nii_im_new = nib.Nifti1Image(data_im_crop, affine_im_new)
        nib.save(nii_im_new, fname_im)
        nii_seg_new = nib.Nifti1Image(data_seg_crop, affine_im_new)
        nib.save(nii_seg_new, fname_seg)


def run_main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Prepare data.')
    parser.add_argument('action',
                        choices=['remove-slice'],
                        )
    parser.add_argument('-i', '--image',
                        help='image in nifti format',
                        type=str,
                        )
    parser.add_argument('-s', '--segmentation',
                        help='segmentation in nifti format',
                        type=str,
                        )
    args = parser.parse_args()
    # Select action
    if args.action == 'remove-slice':
        remove_slice(fname_im=args.image, fname_seg=args.segmentation)


if __name__ == "__main__":
    run_main()


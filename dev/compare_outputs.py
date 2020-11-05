from ivadomed import metrics as imed_metrics
import argparse
import os
import nibabel
import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rater-agreement-path", dest='rpath', required=True, help="Path to folder containing.")
    parser.add_argument("-u", "--uncertainty-path", dest='upath', required=True, help="Path to pred_masks folder.")

    return parser


def uncertainty_by_rater():
    paths = ["log_ms_brain_auto-target_suffix=[\'_seg-lesion0\']","log_ms_brain_auto-target_suffix=[\'_seg-lesion1\']","log_ms_brain_auto-target_suffix=[\'_seg-lesion2\']","log_ms_brain_auto-target_suffix=[\'_seg-lesion3\']","log_ms_brain_auto-target_suffix=[\'_seg-lesion4\']","log_ms_brain_auto-target_suffix=[\'_seg-lesion5\']","log_ms_brain_auto-target_suffix=[\'_seg-lesion6\']","log_ms_brain_auto-target_suffix=[\'_seg-lesion7\']"]
    paths = ["log_ms_brain_auto-target_suffix=[\'_majority-center1\']","log_ms_brain_auto-target_suffix=[\'_majority-center2\']","log_ms_brain_auto-target_suffix=[\'_majority-center3\']","log_ms_brain_auto-target_suffix=[\'_majority-global\']"]
    #paths = ["log_gm_auto-target_suffix=[\'_seg-lesion1\']","log_gm_auto-target_suffix=[\'_seg-lesion2\']","log_gm_auto-target_suffix=[\'_seg-lesion3\']","log_gm_auto-target_suffix=[\'_seg-lesion4\']"]
    im = {}
    for path in paths:
        fnames = os.listdir(os.path.join(path,"pred_masks"))
        for fname in fnames:

            if "unc-vox" in fname:

                if path not in im:
                    im[path] = []
                arr = nibabel.load(os.path.join(path,"pred_masks",fname)).get_data()
                im[path].append(np.sum(arr))
    df = pd.DataFrame.from_dict(im)
    df.to_csv('uncertainty_by_rater.csv')
    print(df)

def multi_model_entropy():
    paths = ["log_ms_brain_auto_filter-target_suffix=[\'_seg-lesion1\']","log_ms_brain_auto_filter-target_suffix=[\'_seg-lesion2\']","log_ms_brain_auto_filter-target_suffix=[\'_seg-lesion3\']","log_ms_brain_auto_filter-target_suffix=[\'_seg-lesion4\']","log_ms_brain_auto_filter-target_suffix=[\'_seg-lesion5\']","log_ms_brain_auto_filter-target_suffix=[\'_seg-lesion6\']","log_ms_brain_auto_filter-target_suffix=[\'_seg-lesion7\']"]
    im = {}
    for path in paths:
        fnames = os.listdir(os.path.join(path,"pred_masks"))
        for fname in fnames:

            if "soft" in fname:
                if fname not in im:
                    im[fname] = []
                im[fname].append(nibabel.load(os.path.join(path,"pred_masks",fname)).get_data())

    eps=1e-5
    for key in im:
        print(key)
        mc_data = np.array(im[key])
        # Compute entropy, from run_uncertainty
        unc = np.repeat(np.expand_dims(mc_data, -1), 2, -1)  # n_it, x, y, z, 2
        unc[..., 0] = 1 - unc[..., 1]
        unc = -np.sum(np.mean(unc, 0) * np.log(np.mean(unc, 0) + eps), -1)
        # Clip values to 0
        unc[unc < 0] = 0

        # save uncertainty map
        nib_unc = nibabel.Nifti1Image(unc, nibabel.load(os.path.join(paths[0],"pred_masks",key)).get_affine())
        nibabel.save(nib_unc, os.path.join("./combined_preds",key))

#Rater disagreement
def compute_difference():
    path = "./"
    niftis = os.listdir(path)
    for nifti in niftis:
        im1 = nibabel.load(os.path.join(path,nifti)).get_data()
        print(np.unique(im1))
        print(np.amax(im1))
        im1 /= np.amax(im1)
        print(np.amax(im1))
        #im1[im1 > 0.5] -= 1
        #im1 = np.abs(im1)
        #nibabel.save( nibabel.Nifti1Image(im1, nibabel.load(path + nifti).get_affine()), path + nifti)

def js_divergence():
    parser = get_parser()
    args = parser.parse_args()
    rpath =  args.rpath
    upath =  args.upath
    niftis = os.listdir(upath)
    div = []
    for nifti in niftis:
        #print(nifti)
        if "unc-vox" in nifti:
            #print("not ignored")
            print(nifti)
            im1 = np.squeeze(nibabel.load(os.path.join(upath,nifti)).get_data())
            #print(np.amax(im1))
            fname = nifti
            im2 = np.squeeze(nibabel.load(os.path.join(rpath,fname)).get_data())
            el = imed_metrics.js_divergence(im1,im2)
            print(el)
            div.append(el)



if __name__ == '__main__':
   #compute_difference()
   #js_divergence()
   #multi_model_entropy()
   uncertainty_by_rater()

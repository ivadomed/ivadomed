from ivadomed import metrics as imed_metrics
import argparse
import os

def ignore(input):
    return "soft" in input

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rater-agreement-path", dest='rpath', required=True, help="Path to folder containing.")
    parser.add_argument("-u", "--uncertainty-path", dest='upath', required=True, help="Path to pred_masks folder.")

    parser = get_parser()
    args = parser.parse_args()
    rpath =  args.rpath
    upath =  args.upath
    niftis = os.listdir(rpath)
    div = []
    for nifti in niftis:
        if not ignore(nifti):
            im1 = nibabel.load(nifti).get_data()
            fname = nifti.split("/")[-1]
            im2 = nibabel.load(os.path.join(upath,fname)).get_data()
            el = imed_metrics.js_divergence(im1,im2)
            print(el)
            div.append(el)

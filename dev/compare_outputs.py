from ivadomed import metrics as imed_metrics
import argparse
import os
import nibabel
import numpy as np
def ignore(input):
    return "soft" not in input

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rater-agreement-path", dest='rpath', required=True, help="Path to folder containing.")
    parser.add_argument("-u", "--uncertainty-path", dest='upath', required=True, help="Path to pred_masks folder.")

    return parser


def compute_difference():
    niftis = os.listdir(upath)
    for nifti in niftis:
        im1 = nibabel.load(os.path.join(upath,nifti)).get_data()
        print(np.amax(im1))
        print(np.amin(im1))


def main():
    parser = get_parser()
    args = parser.parse_args()
    rpath =  args.rpath
    upath =  args.upath
    niftis = os.listdir(upath)
    div = []
    for nifti in niftis:
        #print(nifti)
        if not ignore(nifti):
            #print("not ignored")
            print(nifti)
            im1 = np.squeeze(nibabel.load(os.path.join(upath,nifti)).get_data())
            fname = nifti
            im2 = np.squeeze(nibabel.load(os.path.join(rpath,fname)).get_data())
            el = imed_metrics.js_divergence(im1,im2)
            print(el)
            div.append(el)



if __name__ == '__main__':
    main()

import argparse
import json
import os
import shutil

import nibabel as nib
import numpy as np
import pandas as pd

from ivadomed import config_manager as imed_config_manager
from ivadomed import main as ivado


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log-directory", dest="logdir", required=True, nargs="+",
                        help="Log directory of trained model.")
    parser.add_argument("-b", "--bids-path", dest="bids",
                        required=True, type=str, help="Bids path where are located the GT.")
    parser.add_argument("-n", "--iterations", default=10, type=int, help="Number of Monte Carlo iterations.")
    parser.add_argument("-o", "--output-path", nargs="+", dest="output_path", required=True,
                        type=str, help="Output directory name without extention to save final csv file. There should "
                                       "be the same number of output files parameters as the number of config files.")
    return parser


def get_results(context):
    pred_mask_path = os.path.join(context["log_directory"], "pred_masks")
    if os.path.exists(pred_mask_path):
        shutil.rmtree(pred_mask_path)

    # RandomAffine will be applied during testing
    if "dataset_type" in context["transformation"]["RandomAffine"]:
        del context["transformation"]["RandomAffine"]["dataset_type"]
    if "scale" in context["transformation"]["RandomAffine"]:
        del context["transformation"]["RandomAffine"]["scale"]
    return ivado.run_command(context, command="test")


def compute_csa(config, df_results, logdir, bids):
    subject_list = list(df_results.index)
    # Create empty columns
    df_results = df_results.assign(csa_pred="", csa_gt="", absolute_csa_diff="", relative_csa_diff="")
    for subject in subject_list:
        # Get GT csa
        gt_path = os.path.join(bids, "derivatives", "labels", subject.split("_")[0],
                               "anat", subject + config["loader_parameters"]["target_suffix"][0] + ".nii.gz")
        os.system(f"sct_process_segmentation  -i {gt_path} -perslice 1 -angle-corr 0 -o csa.csv")
        df = pd.read_csv("csa.csv")
        # Take only the medial slice
        csa_gt = float(df["MEAN(area)"][len(df["MEAN(area)"]) // 2])
        os.remove("csa.csv")

        # Get prediction csa
        pred_path = os.path.join(logdir, "pred_masks", subject + "_pred.nii.gz")
        pred_nii = nib.load(pred_path)
        # Keep only first label to compute csa
        single_label_pred = nib.Nifti1Image(pred_nii.get_fdata()[..., 0], pred_nii.affine)
        single_label_pred_path = "pred_single_label.nii.gz"
        nib.save(single_label_pred, single_label_pred_path)
        os.system(
            f"sct_process_segmentation -i {single_label_pred_path} -perslice 1 -angle-corr 0 -o csa.csv")
        df = pd.read_csv("csa.csv")
        # Take only the medial slice
        csa_pred = float(df["MEAN(area)"][len(df["MEAN(area)"]) // 2])

        # Remove files
        os.remove("csa.csv")
        os.system(f"rm {single_label_pred_path}")

        # Populate df with csa stats
        df_results.at[subject, 'csa_pred'] = csa_pred
        df_results.at[subject, 'csa_gt'] = csa_gt
        df_results.at[subject, 'absolute_csa_diff'] = abs(csa_gt - csa_pred) / csa_gt
        df_results.at[subject, 'relative_csa_diff'] = (csa_gt - csa_pred) / csa_gt

    return df_results


def main():
    parser = get_parser()
    args = parser.parse_args()

    for logdir, output_path in zip(args.logdir, args.output_path):
        config = os.path.join(logdir, "config_file.json")
        context = imed_config_manager.ConfigurationManager(config).get_config()

        df_list = []
        metrics = []
        for i in range(int(args.iterations)):
            df = get_results(context)
            df = compute_csa(context, df, logdir, args.bids)
            metrics = list(df.columns)
            df_list.append(np.array(df))

        # Get average and std for each subject (intra subject), then average on all subjects
        average = np.average(np.average(np.array(df_list), axis=0), axis=0)
        std = np.average(np.std(np.array(df_list, dtype=np.float), axis=0), axis=0)
        pd.DataFrame(np.stack([average, std], axis=1), index=metrics, columns=["mean", "std"]).to_csv(output_path +
                                                                                                      ".csv")


if __name__ == '__main__':
    main()

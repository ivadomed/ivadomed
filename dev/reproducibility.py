import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from ivadomed import main as ivado


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Base config file path.")
    parser.add_argument("-n", "--iterations", default=10, type=int, help="Number of Monte Carlo iterations.")
    parser.add_argument("-o", "--output-path", dest="output_path", default="output_reproducibility.csv", type=str,
                        help="Output directory to save final csv file.")
    return parser


def get_results(config):
    with open(config, "r") as fhandle:
        context = json.load(fhandle)

    context["command"] = "eval"
    pred_mask_path = os.path.join(context["log_directory"], "pred_masks")
    if os.path.exists(pred_mask_path):
        shutil.rmtree(pred_mask_path)
    # RandomAffine will be applied during testing
    del context["transformation"]["RandomAffine"]["dataset_type"]
    return ivado.run_command(context)


def compute_csa(config, df_results):
    subject_list = list(df_results.index)
    df_results = df_results.assign(csa_pred="", csa_gt="", absolute_csa_diff="", relative_csa_diff="")
    for subject in subject_list:
        # Get GT csa
        gt_path = os.path.join(config["loader_parameters"]["bids_path"], "derivatives", "labels", subject, "anat",
                               subject + config["loader_parameters"]["target_suffix"][0] + "nii.gz")
        os.system(f"sct_process_segmentation  -i {gt_path} -append 1 -perslice 0 -o csa.csv")
        df = pd.read_csv("csa.csv")
        csa_gt = df["MEAN(area)"]

        # Get prediction csa
        pred_path = os.path.join(config["log_directory"], "pred_masks", subject + "_pred.nii.gz")
        os.system(f"sct_process_segmentation  -i {pred_path} -append 1 -perslice 0 -o csa.csv")
        df = pd.read_csv("csa.csv")
        csa_pred = df["MEAN(area)"]

        # Remove file
        os.system("rm csa.csv")

        # Populate df with csa stats
        df_results['csa_pred'][subject] = csa_pred
        df_results['csa_gt'][subject] = csa_gt
        df_results['absolute_csa_diff'][subject] = abs(csa_gt - csa_pred)
        df_results['relative_csa_diff'][subject] = csa_gt - csa_pred

    return df_results


def main():
    parser = get_parser()
    args = parser.parse_args()

    df_list = []
    subject_list = []
    metrics = []
    for i in range(int(args.iterations)):
        df = get_results(args.config)
        metrics = list(df.columns)
        compute_csa(args.config, df)
        df_list.append(np.array(df))

    # Get average and std for each subject (intra subject), then average on all subjects
    average = np.average(np.average(np.array(df_list), axis=0), axis=0)
    std = np.average(np.std(np.array(df_list), axis=0), axis=0)
    pd.DataFrame(np.stack([average, std], axis=1), index=metrics, columns=["mean", "std"]).to_csv(args.output_path)


if __name__ == '__main__':
    main()

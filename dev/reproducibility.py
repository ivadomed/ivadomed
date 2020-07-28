import argparse
import json
import shutil
import pandas as pd
import numpy as np
import os

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


def main():
    parser = get_parser()
    args = parser.parse_args()

    df_list = []
    metrics = []
    for i in range(int(args.iterations)):
        df = get_results(args.config)
        metrics = list(df.columns)
        df_list.append(np.array(df))

    # Get average and std for each subject (intra subject), then average on all subjects
    average = np.average(np.average(np.array(df_list), axis=0), axis=0)
    std = np.average(np.std(np.array(df_list), axis=0), axis=0)
    pd.DataFrame(np.stack([average, std], axis=1), index=metrics, columns=["mean", "std"]).to_csv(args.output_path)


if __name__ == '__main__':
    main()

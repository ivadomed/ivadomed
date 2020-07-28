import argparse
import json
import shutil
import pandas as pd
import os

from ivadomed import main as ivado


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Base config file path.")
    parser.add_argument("-b", "--bids-path", required=True, help="Bids path where the GT are located.")
    parser.add_argument("-n", "--iterations", default=10, type=int, help="Number of Monte Carlo iterations.")
    parser.add_argument("-o", "--output-path", default="output_reproducibility.csv", type=str,
                        help="Output directory to save final csv file.")
    return parser


def reproducibility_pipeline(config):
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
    for i in range(int(args.iterations)):
        df = reproducibility_pipeline(args.config)
        df_list.append(df)


if __name__ == '__main__':
    main()

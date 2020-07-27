import argparse
import json

from ivadomed import transforms as imed_transforms
from ivadomed.loader import utils as imed_loader_utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Base config file path.")
    return parser


def reproducibility_pipeline(config):
    with open(config, "r") as fhandle:
        context = json.load(fhandle)

    train_lst, valid_lst, test_lst = imed_loader_utils.get_subdatasets_subjects_list(context["split_dataset"],
                                                                                     context['loader_parameters']
                                                                                     ['bids_path'],
                                                                                     context["log_directory"])
    imed_transforms.RandomAffine(degrees=5, scale=[0.1, 0.1], translate=[0.03, 0.03])


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Run automate training
    reproducibility_pipeline(args.config)


if __name__ == '__main__':
    main()

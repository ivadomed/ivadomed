##############################################################
#
# This scripts compute the class balance in a given dataset.
#
# Usage: python dev/class_balance.py -c <config_file_path>
#
# Example: python class_balance.py -c config/config.json
#
# Contributors: charley
# Last modified: 16-10-2019
#
##############################################################

import os
import json
import argparse
import numpy as np

from ivadomed import loader as loader
from ivadomed.utils import SliceFilter

from torchvision import transforms
from torch.utils.data import DataLoader
from medicaltorch import transforms as mt_transforms
from medicaltorch import datasets as mt_datasets


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Config file path.")

    return parser


def print_stats(arr):
    print('\tMean: {}'.format(np.mean(arr)))
    print('\tMedian: {}'.format(np.median(arr)))
    print('\tInter-quartile range: [{}, {}]'.format(np.percentile(arr, 25), np.percentile(arr, 75)))


def run_main(args):

    with open(args.c, "r") as fhandle:
        context = json.load(fhandle)

    transform_lst = transforms.Compose([
        mt_transforms.Resample(wspace=0.75, hspace=0.75),
        mt_transforms.CenterCrop2D((128, 128)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    train_lst, valid_lst, test_lst = loader.split_dataset(context["bids_path"], context["center_test"], context["random_seed"])

    balance_dct = {}
    for ds_lst, ds_name in zip([train_lst, valid_lst, test_lst], ['train', 'valid', 'test']):
        print("\nLoading {} set.\n".format(ds_name))
        ds = loader.BidsDataset(context["bids_path"],
                                 subject_lst=ds_lst,
                                 gt_suffix=context["gt_suffix"],
                                 contrast_lst=context["contrast_test"],
                                 metadata_choice=context["metadata"],
                                 contrast_balance=context["contrast_balance"],
                                 transform=transform_lst,
                                 slice_filter_fn=SliceFilter())

        print("Loaded {} axial slices for the {} set.".format(len(ds), ds_name))
        ds_loader = DataLoader(ds, batch_size=1,
                             shuffle=True, pin_memory=True,
                             collate_fn=mt_datasets.mt_collate,
                             num_workers=1)

        balance_lst = []
        for i, batch in enumerate(ds_loader):
            gt_sample = batch["gt"].numpy().astype(np.int)
            nb_ones = (gt_sample == 1).sum()
            nb_voxels = gt_sample.size
            balance_lst.append(nb_ones * 1.0 / nb_voxels)

        balance_dct[ds_name] = balance_lst

    for ds_name in balance_dct:
        print('\nBalance class in {} set:'.format(ds_name))
        print_stats(balance_dct[ds_name])

    print('\nBalance class in full set:')
    print_stats([e for d in balance_dct for e in balance_dct[d]])

if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    run_main(arguments)

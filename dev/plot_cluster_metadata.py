# Usage:
#	python dev/plot_cluster_metadata.py <config_file>
# Example:
#	python dev/plot_cluster_metadata.py config/config_pr28.json

import sys
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from itertools import groupby
from torchvision import transforms
import matplotlib.pyplot as plt

from ivadomed import loader as loader
from ivadomed.main import SliceFilter
from medicaltorch import transforms as mt_transforms

metadata_type = ['FlipAngle', 'EchoTime', 'RepetitionTime']
metadata_range = {'FlipAngle': [0, 180, 0.5], 'EchoTime': [10**(-3), 10**(0), 10**(-3)], 'RepetitionTime': [10**(-3), 10**(1), 10**(-2)]}


def plot_decision_boundaries(data, model, x_range, metadata_name, fname_out):
    fig = plt.figure()

    x_min, x_max = x_range[0], x_range[1]
    y_min, y_max = 0, (x_max - x_min) * 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_range[2]),
                     np.arange(y_min, y_max, x_range[2]))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.jet, alpha=0.8)

    for s, y_val in zip(['train', 'validation', 'test'], [0.25, 0.5, 0.75]):
        plt.scatter(data[s][metadata_name], [(y_max-y_min) * y_val for v in data[s][metadata_name]], c='k')

    plt.xlabel(metadata_name)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.yticks(())
    if metadata_name != 'FlipAngle':
        plt.xscale('log')

    fig.savefig(fname_out)
    print('\tSave as: '+fname_out)

def run_main(context):

    no_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((128, 128)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    out_dir = context["log_directory"]
    metadata_dct = {}
    for subset in ['train', 'validation', 'test']:
        metadata_dct[subset] = {}
        for bids_ds in tqdm(context["bids_path_"+subset], desc="Loading "+subset+" set"):
            ds = loader.BidsDataset(bids_ds,
                                  contrast_lst=context["contrast_train_validation"] if subset != "test" else context["contrast_test"],
                                  transform=no_transform,
                                  slice_filter_fn=SliceFilter())

            for m in metadata_type:
                if m in metadata_dct:
                    metadata_dct[subset][m] = [v for m_lst in [metadata_dct[subset][m], ds.metadata[m]] for v in m_lst]
                else:
                    metadata_dct[subset][m] = ds.metadata[m]
    # pickle.dump(metadata_dct, open("dev_metadata.pkl", 'wb'))
    # metadata_dct = pickle.load(open("dev_metadata.pkl", "rb"))

    cluster_dct = pickle.load(open(os.path.join(out_dir, "clustering_models.pkl"), "rb"))

    out_dir = os.path.join(out_dir, "cluster_metadata")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for m in metadata_type:
        values = [v for s in ['train', 'validation', 'test'] for v in metadata_dct[s][m]]
        print('\n{}: Min={}, Max={}, Median={}'.format(m, min(values), max(values), np.median(values)))
        plot_decision_boundaries(metadata_dct, cluster_dct[m], metadata_range[m], m, os.path.join(out_dir, m+'.png'))

if __name__ == "__main__":
    fname_config_file = sys.argv[1]

    with open(fname_config_file, "r") as fhandle:
        context = json.load(fhandle)

    run_main(context)

import sys
import os
import json
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt

from ivadomed import loader as loader
from ivadomed.main import SliceFilter
from medicaltorch import transforms as mt_transforms

metadata_type = ['FlipAngle', 'EchoTime', 'RepetitionTime']


def plot_hist(data, fname_out):
    fig = plt.figure()
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    fig.savefig(fname_out)

def run_main(context):

    no_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((128, 128)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    out_dir = context["log_directory"]
    for subset in ['train', 'validation', 'test']:
        metadata_dct = {}
        for bids_ds in tqdm(context["bids_path_"+subset], desc="Loading "+subset+" set"):
            ds = loader.BidsDataset(bids_ds,
                                  transform=no_transform,
                                  slice_filter_fn=SliceFilter())
            for m in metadata_type:
                if m in metadata_dct:
                    metadata_dct[m] = [v for m_lst in [metadata_dct[m], ds.metadata[m]] for v in m_lst]
                else:
                    metadata_dct[m] = ds.metadata[m]

        out_dir_subset = os.path.join(out_dir, subset)
        if not os.path.isdir(out_dir_subset):
            os.makedirs(out_dir_subset)
        for m in metadata_type:
            plot_hist(metadata_dct[m], os.path.join(out_dir_subset, m+'.png'))

if __name__ == "__main__":
    fname_config_file = sys.argv[1]

    with open(fname_config_file, "r") as fhandle:
        context = json.load(fhandle)

    run_main(context)

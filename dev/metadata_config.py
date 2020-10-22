#!/usr/bin/env python
import sys
import json
from tqdm import tqdm
from torchvision import transforms as torch_transforms

from ivadomed import config_manager as imed_config_manager
from ivadomed.loader import loader as imed_loader
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils

metadata_type = ['FlipAngle', 'EchoTime', 'RepetitionTime']


def run_main(context):
    no_transform = torch_transforms.Compose([
        imed_transforms.CenterCrop([128, 128]),
        imed_transforms.NumpyToTensor(),
        imed_transforms.NormalizeInstance(),
    ])

    out_dir = context["log_directory"]
    metadata_dct = {}
    for subset in ['train', 'validation', 'test']:
        metadata_dct[subset] = {}
        for bids_ds in tqdm(context["bids_path_" + subset], desc="Loading " + subset + " set"):
            ds = imed_loader.BidsDataset(bids_ds,
                                         contrast_lst=context["contrast_train_validation"]
                                         if subset != "test" else context["contrast_test"],
                                         transform=no_transform,
                                         slice_filter_fn=imed_utils.SliceFilter())

            for m in metadata_type:
                if m in metadata_dct:
                    metadata_dct[subset][m] = [v for m_lst in [metadata_dct[subset][m], ds.metadata[m]] for v in m_lst]
                else:
                    metadata_dct[subset][m] = ds.metadata[m]

        for m in metadata_type:
            metadata_dct[subset][m] = list(set(metadata_dct[subset][m]))

    with open(out_dir + "/metadata_config.json", 'w') as fp:
        json.dump(metadata_dct, fp)

    return


if __name__ == "__main__":
    fname_config_file = sys.argv[1]

    context = imed_config_manager.ConfigurationManager(fname_config_file).get_config()

    run_main(context)

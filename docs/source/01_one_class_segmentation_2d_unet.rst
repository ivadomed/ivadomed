01. One-class segmentation with 2D U-Net
========================================

This tutorial illustrates the following features:
- XX, XX, XX

Download dataset
-----------------

# TODO

Create and fill your configuration file
----------------------------------------
Examples of configuration files are available in the `ivadomed/config/` folder and the parameter documentation is
available in :doc:`configuration_file`.

We are highlighting here below some key parameters to perform a one-class 2D segmentation training:
- To run a training, use the parameter `command`::

    "command": "train"

- Indicate the location of your BIDS dataset with `bids_path`::

    "bids_path": "path/to/bids/dataset"

- List the target structure by indicating the suffix of its mask in the `derivatives/labels` folder. For a one-class segmentation with our example dataset::

    "target_suffix": ["_seg-manual"]

- Specify the contrast(s) of interest::

    "contrast_params": {
        "training_validation": ["T1w", "T2w"],
        "testing": ["T1w", "T2w"],
        "balance": {}
    }
- Indicate the 2D slice orientation::

    "slice_axis": "axial"

- To perform a multi-channel training (i.e. each sample has several channels, where each channel is an image contrast), then set `multichannel` to `true`. Otherwise, only one image contrast is used per sample. Note: the multichannel approach requires the different image contrasts to be registered together. In this tutorial, only one channel will be used::

    "multichannel": false

Run the training
----------------
Once the configuration file is
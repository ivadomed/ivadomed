One-class segmentation with 2D U-Net
=====================================

Download dataset
-----------------

TODO

Create and fill your configuration file
----------------------------------------
Examples of configuration files are available in the `ivadomed/config/` folder and the parameter documentation is
available in :doc:`configuration_file`.

In particular, make sure to complete the following parameters:
- To run a training, use the parameter `command`::

    "command": "train"

- Indicate the location of your BIDS dataset with `bids_path`::

    "bids_path": "path/to/bids/dataset"

- List the target structures by indicating the suffix of their masks in the `derivatives/labels` folder. For a one-class segmentation with our example dataset::

    "target_suffix": ["_seg-manual"]

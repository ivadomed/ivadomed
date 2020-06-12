..  scripts:

Utility scripts
===============

Visualize data augmentation transformations
***********************************************
Data augmentation is a key part of the Deep Learning training scheme. This script aims at facilitating the fine-tuning of data augmentation parameters. To do so, this script provides a step-by-step visualization of the transformations that are applied on data.

This script applies a series of transformations to 2D slices extracted from an input image, and save as png the resulting sample after each transform.

This is a simple example::

    python dev/visualize_transforms.py -i t2s.nii.gz -n 1 -c config.json -r t2s_seg.nii.gz

Provides a visualisation of a series of three transformation on a randomly selected slice:

.. image:: ../../images/transforms_im.png
    :width: 200px
    :align: center
    :height: 100px

And on a binary mask::

    python dev/visualize_transforms.py -i t2s_gmseg.nii.gz -n 1 -c config.json -r t2s_seg.nii.gz

Gives:

.. image:: ../../images/transforms_gt.png
    :width: 200px
    :align: center
    :height: 100px
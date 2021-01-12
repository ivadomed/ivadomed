
.. _command-line-tools:

Command-Line Tools
##################

.. contents::
   :local:
   :depth: 1
..


Models
******

New model can be generated using the command-line tool from the
terminal:

::

    ivadomed -c path/to/config_file.json

where ``config_file.json`` is a configuration file, which parameters are
described in the :ref:`configuration_file:Configuration File`.

Please see section ``TUTORIALS`` to run this command on an example dataset.

ivadomed
========

.. program-output:: ivadomed -h



Downloading Datasets
********************

We have several datasets hosted in the ``ivadomed`` GitHub organization, which are available
for download. You will see this mentioned in the ``TUTORIALS`` section as well.

::

    ivadomed_download_data -d DATASET_NAME


ivadomed_download_data
======================

.. program-output:: ivadomed_download_data -h


Other Command Line Tools
************************

ivadomed_prepare_dataset_vertebral_labeling
===========================================

.. program-output:: ivadomed_prepare_dataset_vertebral_labeling -h

ivadomed_automate_training
==========================

.. program-output:: ivadomed_automate_training -h

ivadomed_compare_models
=======================

.. program-output:: ivadomed_compare_models -h

ivadomed_visualize_transforms
=============================

.. program-output:: ivadomed_visualize_transforms -h

ivadomed_convert_to_onnx
========================

.. program-output:: ivadomed_convert_to_onnx -h

ivadomed_extract_small_dataset
==============================

.. program-output:: ivadomed_extract_small_dataset -h

ivadomed_training_curve
=======================

.. program-output:: ivadomed_training_curve -h

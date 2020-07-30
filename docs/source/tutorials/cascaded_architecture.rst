Cascaded architecture
=====================

In this tutorial we will learn the following features:

- Training with a cascaded architecture composed of the following two models: 
    - Spinal cord localization model
    - Cerebrospinal fluid (CSF) segmentation

Prerequisite
------------

In this tutorial, the spinal cord segmentation model generated from :doc:`../tutorials/one_class_segmentation_2d_unet`
will be needed.



Configuration file
------------------

In ``ivadomed``, training is orchestrated by a configuration file. Examples of configuration files are available in
the ``ivadomed/config/`` and the documentation is available in :doc:`../configuration_file`.

In this tutorial we will use the configuration file: ``ivadomed/config/config.json``.
First off, copy this configuration file in your local directory (to avoid modifying the source file):

.. code-block:: bash

   cp <PATH_TO_IVADOMED>/ivadomed/config/config.json .

Then, open it with a text editor. Below we will discuss some of the key parameters to perform a one-class 2D
segmentation training.

- ``command``: Action to perform. Here, we want to train a model, so we set the fields as follows:

  .. code-block:: xml

     "command": "train"

- ``loader_parameters:bids_path``: Location of the dataset. As discussed in :doc:`../data`, the dataset
  should conform to the BIDS standard.

  .. code-block:: xml

     "bids_path": "data_example_spinegeneric",

- ``loader_parameters:target_suffix``: Suffix of the ground truth segmentation. The ground truth is located
  under the ``DATASET/derivatives/labels`` folder. In our case, the suffix is ``_seg-manual``:

  .. code-block:: xml

     "target_suffix": ["_seg-manual"]

- ``loader_parameters:contrast_params``: Contrast(s) of interest

  .. code-block:: xml

     "contrast_params": {
         "training_validation": ["T1w", "T2w", "T2star"],
         "testing": ["T1w", "T2w", "T2star"],
         "balance": {}
     }

- ``loader_parameters:slice_axis``: Orientation of the 2D slice to use with the model.

  .. code-block:: xml

     "slice_axis": "axial"

- ``loader_parameters:multichannel``: Turn on/off multi-channel training. If ``true``, each sample has several
  channels, where each channel is an image contrast. If ``false``, only one image contrast is used per sample.

  .. code-block:: xml

     "multichannel": false

  .. note::

     The multichannel approach requires that for each subject, the image contrasts are co-registered. This implies that
     a ground truth segmentation is aligned with all contrasts, for a given subject. In this tutorial, only one channel
     will be used.


Train model
-----------

Once the configuration file is ready, run the training:

.. code-block:: bash

   ivadomed config.json

.. note::

   If a `compatible GPU <https://pytorch.org/get-started/locally/>`_ is available, it will be used by default. Otherwise, training will use the CPU, which will take
   a prohibitively long computational time (several hours).

The main parameters of the training scheme and model will be displayed on the terminal, followed by the loss value
on training and validation sets at every epoch. To know more about the meaning of each parameter, go to
:doc:`../configuration_file`. The value of the loss should decrease during the training.

.. code-block:: console

   Creating log directory: spineGeneric
   Using GPU number 0

   Selected transformations for the training dataset:
   Resample: {'wspace': 0.75, 'hspace': 0.75, 'dspace': 1, 'preprocessing': True}
   CenterCrop: {'size': [128, 128], 'preprocessing': True}
   RandomAffine: {'degrees': 5, 'scale': [0.1, 0.1], 'translate': [0.03, 0.03], 'applied_to': ['im', 'gt']}
   ElasticTransform: {'alpha_range': [28.0, 30.0], 'sigma_range': [3.5, 4.5], 'p': 0.1, 'applied_to': ['im', 'gt']}
   NumpyToTensor: {}
   NormalizeInstance: {'applied_to': ['im']}

   Selected transformations for the validation dataset:
   Resample: {'wspace': 0.75, 'hspace': 0.75, 'dspace': 1, 'preprocessing': True}
   CenterCrop: {'size': [128, 128], 'preprocessing': True}
   NumpyToTensor: {}
   NormalizeInstance: {'applied_to': ['im']}

   Selected architecture: Unet, with the following parameters:
   dropout_rate: 0.3
   bn_momentum: 0.9
   depth: 4
   folder_name: seg_sc_t1_t2_t2s_mt
   in_channel: 1
   out_channel: 1
   Loading dataset: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 1854.79it/s]
   Loaded 93 axial slices for the validation set.
   Loading dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 1815.06it/s]
   Loaded 291 axial slices for the training set.
   Creating model directory: spineGeneric/seg_sc_t1_t2_t2s_mt

   Initialising model's weights from scratch.

   Scheduler parameters: {'base_lr': 1e-05, 'max_lr': 0.01}

   Selected Loss: DiceLoss
   with the parameters: []
   Epoch 1 training loss: -0.0420.
   Epoch 1 validation loss: -0.0507.

After 100 epochs (see ``"num_epochs"`` in the configuration file), the Dice score on the validation set should
be ~90%.

Evaluate model
--------------

To test the trained model on the testing sub-dataset and compute evaluation metrics, open your config file and
set ``command`` to ``eval``:

.. code-block:: bash

   "command": "eval"

Then run:

.. code-block:: bash

   ivadomed config.json

The model's parameters will be displayed in the terminal, followed by a preview of the results for each image.
The resulting segmentation is saved for each image in the ``<log_directory>/pred_masks`` while a csv file,
saved in ``log_directory/results/eval/evaluation_3Dmetrics.csv``, contains all the evaluation metrics. For more details
on the evaluation metrics, see :mod:`ivadomed.metrics`.

.. code-block:: console

   Log directory already exists: spineGeneric
   Using GPU number 0

   Selected architecture: Unet, with the following parameters:
   dropout_rate: 0.3
   bn_momentum: 0.9
   depth: 4
   folder_name: seg_sc_t1_t2_t2s_mt
   in_channel: 1
   out_channel: 1

   Run Evaluation on spineGeneric/pred_masks

   Evaluation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:06<00:00,  1.33s/it]
                             avd_class0  dice_class0  lfdr_101-INFvox_class0  lfdr_class0          ...            specificity_class0  vol_gt_class0  vol_pred_class0  lfdr_21-100vox_class0
   image_id                                                                                       ...
   sub-strasbourg04_T2w       0.047510     0.921796                     0.0          0.0          ...                      0.999939         4920.0          4686.25                    NaN
   sub-hamburg01_T2w          0.013496     0.943535                     0.0          0.0          ...                      0.999934         5650.0          5573.75                    NaN
   sub-hamburg01_T1w          0.103540     0.902706                     0.0          0.0          ...                      0.999946         5650.0          5065.00                    NaN
   sub-strasbourg04_T2star    0.082561     0.917791                     0.0          0.0          ...                      0.999852         4315.0          4671.25                    NaN
   sub-strasbourg04_T1w       0.437246     0.697122                     0.5          0.5          ...                      0.999979         4920.0          2768.75                    NaN

   [5 rows x 16 columns]


The test image segmentations are stored in ``<log_directory>/pred_masks/`` and have the same name as the input image
with the suffix ``_pred``. To visualize the segmentation of a given subject, you can use any Nifti image viewer.
For `FSLeyes <https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/userdoc/latest/>`_ user, this command will open the
input image with the overlaid prediction (segmentation):

.. code-block:: bash

   fsleyes path/to/input/image.nii.gz path/to/pred_masks/subject_id_contrast_pred.nii.gz -cm red -a 0.5

After the training for 100 epochs, the segmentations should be similar to the one presented in the following image.
The output and ground truth segmentations of the spinal cord are presented in red (subject ``sub-hamburg01`` with
contrast T2w):

.. image:: ../../../images/sc_prediction.png
   :align: center

Two-classes microscopy segmentation with 2D U-Net
=================================================

In this tutorial we will learn the following features:

- Training of a segmentation model (U-Net 2D) with a two classes labels on a single contrast on microscopy PNG images,

- Testing of a trained model and computation of 3D evaluation metrics,

- Visualization of the outputs of a trained model.

.. _Download dataset:

Download dataset
----------------

We will use a publicly-available dataset consisting of 10 microscopy sampel of rat spinal cord.

To download the dataset (~11MB), run the following commands in your terminal:

.. code-block:: bash

   # Download data
   ivadomed_download_data -d data_axondeepseg_sem

Configuration file
------------------

In ``ivadomed``, training is orchestrated by a configuration file. Examples of configuration files are available in
the ``ivadomed/config/`` and the documentation is available in :doc:`../configuration_file`.

In this tutorial we will use the configuration file: ``ivadomed/config/config_microscopy.json``.
First off, copy this configuration file in your local directory (to avoid modifying the source file):

.. code-block:: bash

   cp <PATH_TO_IVADOMED>/ivadomed/config/config_microscopy.json .

Then, open it with a text editor. Below we will discuss some of the key parameters to perform a two-class 2D
microscopy segmentation training.

- ``command``: Action to perform. Here, we want to train a model, so we set the fields as follows:

  .. code-block:: xml

     "command": "train"

- ``path_output``: Folder name that will contain the output files (e.g., trained model, predictions, results).

  .. code-block:: xml

     "path_output": "log_microscopy_sem"

- ``loader_parameters:path_data``: Location of the dataset. As discussed in `Data <../data.html>`__, the dataset
  should conform to the BIDS standard. Modify the path so it points to the location of the downloaded dataset.

  .. code-block:: xml

     "path_data": "data_axondeepseg_sem"

- ``loader_parameters:bids_config``: Location of the custom bids configuration file required for microscopy file indexing.

  .. code-block:: xml

     "bids_config": "ivadomed/config/config_bids.json",

- ``loader_parameters:target_suffix``: Suffix of the ground truths segmentation. The ground truths are located
  under the ``data_axondeepseg_sem/derivatives/labels`` folder. In our case, the suffix are ``_seg-axon-manual`` and ``_seg-myelin-manual``:

  .. code-block:: xml

     "target_suffix": ["_seg-axon-manual", "_seg-myelin-manual"]

- ``loader_parameters:extensions``: File extensions of the microscopy raw data

  .. code-block:: xml

     "extensions": [".png"]

- ``loader_parameters:contrast_params``: Contrast of interest

  .. code-block:: xml

     "contrast_params": {
         "training_validation": ["SEM"],
         "testing": ["SEM"],
         "balance": {}
     }

- ``loader_parameters:slice_axis``: Orientation of the 2D slice to use with the model.
  2D PNG files must use default ``axial``.

  .. code-block:: xml

     "slice_axis": "axial"

- ``split_dataset:split_method``: Describe the metadata used to split the train/validation/test sets.
  Here, ``sample_id`` from the ``samples.tsv`` file will shuffle all sampless, then split between train/validation/test sets.
- ``split_dataset:train_fraction``: Fraction of the dataset's ``sample_id`` in the train set.
- ``split_dataset:test_fraction``: Fraction of the dataset's ``sample_id`` in the test set.

  .. code-block:: xml

      "split_method" : "sample_id"
      "train_fraction": 0.6
      "test_fraction": 0.1

- ``training_time:num_epochs``: the maximum number of epochs that will be run during training. Each epoch is composed
  of a training part and an evaluation part. It should be a strictly positive integer. For this tutorial, we will use
  50 epochs.

  .. code-block:: xml

     "num_epochs": 50

- ``default_model:depth``: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
- ``default_model:length_2D``: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
- ``default_model:stride_2D``: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

  .. code-block:: xml

     "depth": 3
     "length_2D": [256, 256]
     "stride_2D": [244, 244]

- ``postprocessing:binarize_maxpooling``: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

  .. code-block:: xml

      "binarize_maxpooling": {}

- ``transformation:Resample``: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

  .. code-block:: xml

     "Resample":
        {
            "hspace": 0.0001,
            "wspace": 0.0001
        },


Train model
-----------

Once the configuration file is ready, run the training:

.. code-block:: bash

   ivadomed -c config_microscopy.json

Alternatively, the "command", "path_output", and "path_data" arguments can be passed as CLI flags
in which case they supersede the configration file, see `Usage <../usage.html>`__.

.. code-block:: bash

   ivadomed --train -c config_microscopy.json --path-data path/to/bids/data --path-output path/to/output/directory

.. note::

   If a `compatible GPU <https://pytorch.org/get-started/locally/>`_ is available, it will be used by default.
   Otherwise, training will use the CPU, which will take a prohibitively long computational time (several hours).

The main parameters of the training scheme and model will be displayed on the terminal, followed by the loss value
on training and validation sets at every epoch. To know more about the meaning of each parameter, go to
:doc:`../configuration_file`. The value of the loss should decrease during the training.

.. code-block:: console

   Creating output path: log_microscopy_sem
   Using GPU ID 0

   Selected architecture: Unet, with the following parameters:
   dropout_rate: 0.2
   bn_momentum: 0.1
   depth: 3
   is_2d: True
   final_activation: sigmoid
   length_2D: [256, 256]
   stride_2D: [244, 244]
   folder_name: model_seg_rat_axon-myelin_sem
   in_channel: 1
   out_channel: 3
   Dataframe has been saved in log_microscopy_sem/bids_dataframe.csv.
   After splitting: train, validation and test fractions are respectively 0.6, 0.3 and 0.1 of sample_id.

   Selected transformations for the ['training'] dataset:
   Resample: {'hspace': 0.0001, 'wspace': 0.0001}
   RandomAffine: {'degrees': 2.5, 'scale': [0.05, 0.05], 'translate': [0.015, 0.015], 'applied_to': ['im', 'gt']}
   ElasticTransform: {'alpha_range': [100.0, 150.0], 'sigma_range': [4.0, 5.0], 'p': 0.5, 'applied_to': ['im', 'gt']}
   NormalizeInstance: {'applied_to': ['im']}
   Selected transformations for the ['validation'] dataset:
   Resample: {'hspace': 0.0001, 'wspace': 0.0001}
   NormalizeInstance: {'applied_to': ['im']}

   Loading dataset: 100%████████████████████████████████████████████████████████████████ 3/3 [00:00<00:00, 738.48it/s]
   Loaded 76 axial patches of shape [256, 256] for the validation set.
   Loading dataset: 100%████████████████████████████████████████████████████████████████ 6/6 [00:00<00:00, 829.21it/s]
   Loaded 252 axial patches of shape [256, 256] for the training set.
   Creating model directory: log_microscopy_sem/model_seg_rat_axon-myelin_sem

   Initialising model's weights from scratch.

   Scheduler parameters: {'name': 'CosineAnnealingLR', 'base_lr': 1e-05, 'max_lr': 0.01}

   Selected Loss: DiceLoss
   with the parameters: []
   Epoch 1 training loss: -0.6894.
   Epoch 1 validation loss: -0.7908.

After 50 epochs (see ``"num_epochs"`` in the configuration file), the Dice score on the validation set should be ~85%.

.. note::

   TEMP NIFTI FILE XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

.. _Evaluate model:

Evaluate model
--------------

To test the trained model on the testing sub-dataset and compute evaluation metrics, run:

.. code-block:: bash

   ivadomed --test -c config.json --path-data path/to/bids/data --path-output path/to/output/directory

If you prefer to use config files over CLI flags, set "command" to the following in you config file:
. code-block:: bash

   "command": "test"

You can also set "path_output", and "path_data" arguments in your config file.

Then run:

.. code-block:: bash

   ivadomed -c config.json

The model's parameters will be displayed in the terminal, followed by a preview of the results for each image.
The resulting segmentations are saved for each image in the ``<PATH_TO_OUT_DIR>/pred_masks`` while a csv file,
saved in ``<PATH_TO_OUT_DIR>/results_eval/evaluation_3Dmetrics.csv``, contains all the evaluation metrics. For more details
on the evaluation metrics, see :mod:`ivadomed.metrics`.

.. code-block:: console

   Output path already exists: log_microscopy_sem
   Using GPU ID 0

   Selected architecture: Unet, with the following parameters:
   dropout_rate: 0.2
   bn_momentum: 0.1
   depth: 3
   is_2d: True
   final_activation: sigmoid
   length_2D: [256, 256]
   stride_2D: [244, 244]
   folder_name: model_seg_rat_axon-myelin_sem
   in_channel: 1
   out_channel: 3
   Dataframe has been saved in /home/GRAMES.POLYMTL.CA/maboudb/data_extrassd_maboudb/20211228_tuto/log_microscopy_sem_depth3/bids_dataframe.csv.
   After splitting: train, validation and test fractions are respectively 0.6, 0.3 and 0.1 of sample_id.

   Selected transformations for the ['testing'] dataset:
   Resample: {'hspace': 0.0001, 'wspace': 0.0001}
   NormalizeInstance: {'applied_to': ['im']}

   Loading dataset: 100%████████████████████████████████████████████████████████████████ 1/1 [00:00<00:00, 413.48it/s]
   Loaded 16 axial patches of shape [256, 256] for the testing set.
   Loading model: log_microscopy_sem/best_model.pt

   Inference - Iteration 0: 100%|████████████████████████████████████████████████████████████████ 4/4 [00:01<00:00,  2.89it/s]
   Lossy conversion from float64 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
   Lossy conversion from float64 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
   {'dice_score': 0.845923333420395, 'multi_class_dice_score': 0.8504714482500202, 'precision_score': 0.8343947513053036,
   'recall_score': 0.8577749527693911, 'specificity_score': 0.9447110780829443, 'intersection_over_union': 0.7329871211481128,
   'accuracy_score': 0.9233990207484972, 'hausdorff_score': 0.0}

   Run Evaluation on log_microscopy_sem/pred_masks

   Evaluation: 100%████████████████████████████████████████████████████████████████ 1/1 [00:13<00:00, 13.56s/it]
   Lossy conversion from float64 to uint8. Range [0.0, 3.0]. Convert image to uint8 prior to saving to suppress this warning.
   Lossy conversion from float64 to uint8. Range [0.0, 3.0]. Convert image to uint8 prior to saving to suppress this warning.
                                avd_class0  avd_class1  dice_class0  dice_class1  lfdr_class0  lfdr_class1  ...  specificity_class0  specificity_class1  vol_gt_class0  vol_gt_class1  vol_pred_class0  vol_pred_class1
   image_id                                                                                               ...
   sub-rat3_sample-data9_SEM    0.070742    0.106846      0.87972     0.821223     0.558304     0.319328  ...            0.977565             0.90937   1.256960e-07   1.574890e-07     1.168040e-07     1.743160e-07

   [1 rows x 26 columns]

The test image segmentations are stored in ``<PATH_TO_OUT_DIR>/pred_masks/`` in PNG format and have the same name as the input image
with the suffix ``<target_suffix>_pred``.
A temporary NIfTI files containing the predictions for both classes with the suffix ``_pred`` will also be present.

After the training for 50 epochs, the segmentations should be similar to the one presented in the following image.
The output and ground truth segmentations of the axons and myelin are presented in blue and red respectively
for ``sub-rat3_sample-data9_SEM``):

.. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/tutorials/one_class_segmentation_2d_unet/sc_prediction.png
   :align: center

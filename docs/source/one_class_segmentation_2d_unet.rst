One-class segmentation with 2D U-Net
========================================

This tutorial illustrates the following features:

- Training of a segmentation model (U-Net 2D) with a single label on multiple contrasts.

- Testing of a trained model.

- Visualization of the outputs of a trained model.

Download dataset
-----------------

A sample dataset of spinal cord segmentation is available on `GitHub <https://github.com/ivadomed/data_spinegeneric_registered>`_. Three contrasts are available for each patient: T1w, T2w and T2star. All images are registered.

To download the dataset, perform these command lines in your terminal::

    curl -o ivadomed_spinegeneric_registered.zip -L https://github.com/ivadomed/data_spinegeneric_registered/archive/master.zip

    unzip ivadomed_spinegeneric_registered.zip


Create and fill your configuration file
----------------------------------------
Examples of configuration files are available in the `ivadomed/config/` folder and the parameter documentation is
available in :doc:`configuration_file`.

We are highlighting here below some key parameters to perform a one-class 2D segmentation training:

- To run a training, use the parameter `command`::

    "command": "train"

- Indicate the location of your BIDS dataset with `bids_path`. If you downloaded the sample dataset using the lines mentioned above, the bids path should finish by "data_spinegeneric_registered-master"::

    "bids_path": "path/to/bids/dataset"

- List the target structure by indicating the suffix of its mask in the `derivatives/labels` folder. For a one-class segmentation with our example dataset::

    "target_suffix": ["_seg-manual"]

- Specify the contrast(s) of interest::

    "contrast_params": {
        "training_validation": ["T1w", "T2w", "T2star"],
        "testing": ["T1w", "T2w", "T2star"],
        "balance": {}
    }
- Indicate the 2D slice orientation::

    "slice_axis": "axial"

- To perform a multi-channel training (i.e. each sample has several channels, where each channel is an image contrast), then set `multichannel` to `true`. Otherwise, only one image contrast is used per sample. Note: the multichannel approach requires the different image contrasts to be registered together. In this tutorial, only one channel will be used::

    "multichannel": false

Run the training
----------------
Once the configuration file is filled, you can run the training by launching::

    ivadomed path/to/config/file.json

# TODO: give terminal output and comment.

Evaluate model performance on the testing sub-dataset
-----------------------------------------------------
In order to test the trained model on the testing sub-dataset and compute evaluation metrics, open your config file and set `command` to `eval`. Then run:

    ivadomed path/to/config/file.json

The main parameters of the model will appear, followed by the loss value on training and validation sets at every epoch.

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

The resulting segmentation is saved for each image in the `log_directory/pred_masks` while a csv file, saved in XX, contains all the evaluation metrics.

# TODO: illustrate

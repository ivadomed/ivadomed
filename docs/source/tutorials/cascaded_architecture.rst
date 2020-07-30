Cascaded architecture
=====================

In this tutorial we will learn the following features:

- Training with a cascaded architecture composed of the following two models: 
    - Spinal cord localization model
    - Cerebrospinal fluid (CSF) segmentation
- Visualize training with tensorboard

The model will first locate the spinal cord. Then, the input images will be crop around the spinal cord tumor mask.
Finally, from the cropped images, the CSF will be segmented. The first cropping step allows the final segmentation
model to focus only on the most important part of the image.

.. _Prerequisite:

Prerequisite
------------

In this tutorial, the spinal cord segmentation model generated from :doc:`../tutorials/one_class_segmentation_2d_unet`
will be needed. In the log directory of this trained model, a folder named ``seg_sc_t1_t2_t2s_mt`` (if the default value
of the parameter ``model_name`` was used) contains the packaged model. The path to this folder will be
needed in this tutorial and will be referenced as the location to the object detection model.


Configuration file
------------------

In this tutorial we will use the configuration file: ``ivadomed/config/config.json``.
First off, copy this configuration file in your local directory (to avoid modifying the source file):

.. code-block:: bash

   cp <PATH_TO_IVADOMED>/ivadomed/config/config.json .

Then, open it with a text editor. As described in the tutorial :doc:`../tutorials/one_class_segmentation_2d_unet`, make
sure the ``command`` is set to "train" and ``bids_path`` point to the location of the dataset. Below, we will discuss
some of the key parameters to use cascaded models.

- ``object_detection_params:object_detection_path``: Location of the object detection model. This parameter corresponds
  to the prerequisite model path  to the trained model (see :ref:`Prerequisite`). This spinal cord segmentation model
  will be applied to the images and a bounding box will be created around this mask to crop the image.

  .. code-block:: xml

     "object_detection_path": "<SPINAL_CORD_SEG_LOG_DIRECTORY>/spineGeneric/seg_sc_t1_t2_t2s_mt"

- ``object_detection_params:safety_factor``: Multiplicative factor to apply to each dimension of the bounding box. To
  ensure all the CSF is included, a safety factor should be applied to the bounding box generated from the spinal cord.
  A safety factor of 10% on each dimension is applied here.

  .. code-block:: xml

     "safety_factor": [1.1, 1.1, 1.1]

- ``loader_parameters:target_suffix``: Suffix of the ground truth segmentation. The ground truth are located under the
  ``DATASET/derivatives/labels`` folder. The suffix for CSF labels in this dataset is ``_csfseg-manual``:

  .. code-block:: xml

     "target_suffix": ["_csfseg-manual"]

- ``loader_parameters:contrast_params``: Contrast(s) of interest. The CSF labels are only available in T2w contrast in
  the example dataset.

  .. code-block:: xml

     "contrast_params": {
         "training_validation": ["T2w"],
         "testing": ["T2w"],
         "balance": {}
     }

- ``transformation:CenterCrop``: Crop size in pixel. Images will be cropped or padded to fit these dimensions. This
  allows all the images to have the same size during training. Since the images will be cropped around the spinal cord,
  the image size can be reduced.

  .. code-block:: xml

     "transformation": {
         "CenterCrop": {
             "size": [64, 64],
             "preprocessing": true
         }
     }

Train model
-----------

Once the configuration file is ready, run the training:

.. code-block:: bash

   ivadomed -c config.json

.. note::

   If a `compatible GPU <https://pytorch.org/get-started/locally/>`_ is available, it will be used by default. Otherwise, training will use the CPU, which will take
   a prohibitively long computational time (several hours).

Visualize training data
-----------------------

Tensorboard helps visualize the augmented input images, the model's prediction, the groud truth, the learning curves and
more. To access this data, use the following command-line:

.. code-block:: bash
   tensorboard --logdir <PATH_TO_LOG_DIRECTORY>

The following should be displayed in the terminal:

.. code-block:: console
   Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
   TensorBoard 2.2.1 at http://localhost:6006/ (Press CTRL+C to quit)

Open your browser and type the URL provided, in this case ``http://localhost:6006/``.
In the scalars folder, the evolution of metrics and loss through the epochs can be visualize.

## ADD SCREENSHOT ##

In the image folder, the training and validation ground truth, input images and predictions are displayed. With this
feature, it is possible to visualize the cropping from the first model and confirm that the spinal cord
was correctly located and that the cropping was successful.

## ADD SCREENSHOT ##


Evaluate model
--------------

To test and apply this model the dataset go to the `Evaluate model` section of the tutorial
:doc:`../tutorials/one_class_segmentation_2d_unet`.

Cascaded architecture
=====================

    In this tutorial we will learn the following features:

    - Design a training scheme composed of two cascaded networks.
    - Visualize the training with tensorboard.
    - Generate a GIF to visualize the learning of the model.
    - Find the optimal threshold to binarize images based on the validation sub-dataset.

    In our example, the model will first locate the spinal cord (step 1). This localisation will then be used to crop the images around this region of interest, before segmenting the cerebrospinal fluid (CSF, step 2).

Download dataset
----------------

    A dataset example is available for this tutorial. If not already done, download the dataset with the following line.
    For more details on this dataset see :ref:`One-class segmentation with 2D U-Net<Download dataset>`.

    .. code-block:: bash

       # Download data
       ivadomed_download_data -d data_example_spinegeneric

Configuration file
------------------

    In this tutorial we will use the configuration file: ``ivadomed/config/config.json``.
    First off, copy this configuration file in your local directory to avoid modifying the source file:

    .. code-block:: bash

       cp <PATH_TO_IVADOMED>/ivadomed/config/config.json .

    Then, open it with a text editor. As described in the tutorial :doc:`../tutorials/one_class_segmentation_2d_unet`, make
    sure the ``command`` is set to "train" and ``path_data`` point to the location of the dataset. Below, we will discuss
    some of the key parameters to use cascaded models.

    - ``debugging``: Boolean, create extended verbosity and intermediate outputs. Here we will look at the intermediate predictions
      with tensorboard, we therefore need to activate those intermediate outputs.

      .. code-block:: xml

         "debugging": true

    - ``object_detection_params:object_detection_path``: Location of the object detection model. This parameter corresponds
      to the path of the first model from the cascaded architecture that segments the spinal cord. The packaged model in the
      downloaded dataset located in the folder `trained_model/seg_sc_t1-t2-t2s-mt` will be used to detect the spinal cord.
      This spinal cord segmentation model will be applied to the images and a bounding box will be created around this mask
      to crop the image.

      .. code-block:: xml

         "object_detection_path": "<PATH_TO_DATASET>/trained_model/seg_sc_t1-t2-t2s-mt"

    - ``object_detection_params:safety_factor``: Multiplicative factor to apply to each dimension of the bounding box. To
      ensure all the CSF is included, a safety factor should be applied to the bounding box generated from the spinal cord.
      A safety factor of 200% on each dimension is applied on the height and width of the image. The original depth of the
      bounding box is kept since the CSF should not be present past this border.

      .. code-block:: xml

         "safety_factor": [2, 2, 1]

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

    - ``transformation:CenterCrop:size``: Crop size in voxel. Images will be cropped or padded to fit these dimensions. This
      allows all the images to have the same size during training. Since the images will be cropped around the spinal cord,
      the image size can be reduced to avoid large zero padding.

      .. code-block:: xml

         "CenterCrop": {
             "size": [64, 64]
         }

Train model
-----------

    Once the configuration file is ready, run the training. `ivadomed` has an option to find a threshold value which optimized the dice score on the validation dataset. This threshold will be further used to binarize the predictions on testing data. Add the flag `-t` with an increment
    between 0 and 1 to perform this threshold optimization (i.e. ``-t 0.1`` will return the best threshold between 0.1,
    0.2, ..., 0.9)

    To help visualize the training, the flag ``--gif`` or ``-g`` can be used. The flag should be followed by the number of
    slices by epoch to visualize. For example, ``-g 2`` will generate 2 GIFs of 2 randomly selected slices from the
    validation set.

    Make sure to run the CLI command with the ``--train`` flag, and to point to the location of the dataset via the flag ``--path-data path/to/bids/data``.

    .. code-block:: bash

       ivadomed --train -c config.json --path-data path/to/bids/data --path-output path/to/output/directory -t 0.01 -g 1

    If you prefer to use config files over CLI flags, set ``command`` to the following in you config file:

    .. code-block:: bash

       "command": "train"

    You can also set ``path_output``, and ``path_data`` arguments in your config file.

    Then run:

    .. code-block:: bash

       ivadomed -c config.json

    At the end of the training, the optimal threshold will be indicated:

    .. code-block:: console

       Running threshold analysis to find optimal threshold
        Optimal threshold: 0.01
        Saving plot: spineGeneric/roc.png


Visualize training data
-----------------------
    If the flag ``--gif`` or ``-g`` was used, the training can be visualized through gifs located in the folder specified by the ``--path-output`` flag
    ``<PATH_TO_OUT_DIR>/gifs``.

    .. figure:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/tutorials/cascaded_architecture/training.gif
       :width: 300
       :align: center

       Training visualization with GIF

    Another way to visualize the training is to use Tensorboard. Tensorboard helps to visualize the augmented input images,
    the model's prediction, the ground truth, the learning curves, and more. To access this data during or after training,
    use the following command-line:

    .. code-block:: bash

       tensorboard --logdir <PATH_TO_OUT_DIR>

    The following should be displayed in the terminal:

    .. code-block:: console

       Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
       TensorBoard 2.2.1 at http://localhost:6006/ (Press CTRL+C to quit)

    Open your browser and type the URL provided, in this case ``http://localhost:6006/``.
    In the scalars folder, the evolution of metrics, learning rate and loss through the epochs can be visualized.

    .. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/tutorials/cascaded_architecture/tensorboard_scalar.png
       :align: center

    In the image folder, the training and validation ground truth, input images and predictions are displayed. With this
    feature, it is possible to visualize the cropping from the first model and confirm that the spinal cord
    was correctly located.

    .. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/tutorials/cascaded_architecture/tensorboard_images.png
       :align: center

Evaluate model
--------------
    - ``postprocessing:binarize_prediction``: Threshold at which predictions are binarized. Before testing the model,
      modify the binarization threshold to have a threshold adapted to the data:

    .. code-block:: xml

        "binarize_prediction": 0.01


    To test and apply this model on the testing dataset, go to the `Evaluate model` section of the tutorial
    :ref:`One-class segmentation with 2D U-Net<evaluate model>`.

Pre-trained models
==================

For convenience, the following pre-trained models are ready-to-use:

- `t2-tumor <https://github.com/ivadomed/t2_tumor>`_: Cord tumor segmentation model, trained on T2-weighted contrast.
- `t2star_sc <https://github.com/ivadomed/t2star_sc>`_: Spinal cord segmentation model, trained on T2-star contrast.
- `mice_uqueensland_gm <https://github.com/ivadomed/mice_uqueensland_gm>`_: Gray matter segmentation model on mouse MRI. Data from University of Queensland.
- `mice_uqueensland_sc <https://github.com/ivadomed/mice_uqueensland_sc>`_: Cord segmentation model on mouse MRI. Data from University of Queensland.
- `findcord_tumor <https://github.com/ivadomed/findcord_tumor>`_: Cord localisation model, trained on T2-weighted images with tumor.
- `model_find_disc_t1 <https://github.com/ivadomed/model_find_disc_t1>`_: Intervertebral disc detection model trained on T1-weighted images.
- `model_find_disc_t2 <https://github.com/ivadomed/model_find_disc_t2>`_: Intervertebral disc detection model trained on T2-weighted images.

Packaged model format
---------------------
Each folder contains a model (.pt or .onnx) with its corresponding configuration file (.json). The packaged model is
automatically generated during training. The folder containing the packaged model will be saved at the path specified by
the CLI flag ``--path-output`` or "path_output" in your config file. The packaged model, the configuration file, and the model file will
be named by the string specified by the key ``model_name`` in the configuration file.

.. code-block:: xml

   my_model
   ├── my_model.json
   └── my_model.onnx

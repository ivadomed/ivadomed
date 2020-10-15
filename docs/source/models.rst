Architectures
=============

Implementation
------------------

:mod:`ResNet`
^^^^^^^^^^^^^

.. autoclass:: ivadomed.models.ResNet


:mod:`DenseNet`
^^^^^^^^^^^^^^^

.. autoclass:: ivadomed.models.DenseNet


:mod:`Unet`
^^^^^^^^^^^

.. autoclass:: ivadomed.models.Unet


:mod:`FiLMedUnet`
^^^^^^^^^^^^^^^^^

.. autoclass:: ivadomed.models.FiLMedUnet


:mod:`HeMISUnet`
^^^^^^^^^^^^^^^^

.. autoclass:: ivadomed.models.HeMISUnet


:mod:`UNet3D`
^^^^^^^^^^^^^

.. autoclass:: ivadomed.models.UNet3D


:mod:`Countception`
^^^^^^^^^^^^^^^^^^^

.. autoclass:: ivadomed.models.Countception

Models
------

Different models trained with ivadomed are publicly available:

- `t2-tumor <https://github.com/ivadomed/t2_tumor/archive/r20200621.zip>`_: Cord tumor segmentation model, trained on T2-weighted contrast.
- `t2star_sc <https://github.com/ivadomed/t2star_sc/archive/r20200622.zip>`_: Spinal cord segmentation model, trained on T2-star contrast.
- `mice_uqueensland_gm <https://github.com/ivadomed/mice_uqueensland_gm/archive/r20200622.zip>`_: Gray matter segmentation model on mouse MRI. Data from University of Queensland.
- `mice_uqueensland_sc <https://github.com/ivadomed/mice_uqueensland_sc/archive/r20200622.zip>`_: Cord segmentation model on mouse MRI. Data from University of Queensland.
- `findcord_tumor <https://github.com/ivadomed/findcord_tumor/archive/r20200621.zip>`_: Cord localisation model, trained on T2-weighted images with tumor.
- `model_find_disc_t1 <https://github.com/ivadomed/model_find_disc_t1/archive/r20201013.zip>`_: Intervertebral disc detection model trained on T1-weighted images.
- `model_find_disc_t2 <https://github.com/ivadomed/model_find_disc_t2/archive/r20200928.zip>`_: Intervertebral disc detection model trained on T2-weighted images.
Estimate uncertainty
=====================

This tutorial shows how to estimate uncertainty measures on the model predictions. Uncertainty measures that are
implemented in ``ivadomed`` are detailed :doc:`../technical_features#uncertainty-measures`.

.. _Prerequisite:

Prerequisite
------------

The spinal cord segmentation model generated from :doc:`../tutorials/one_class_segmentation_2d_unet` will be used to
estimate uncertainty at inference time. Please make sure you did this tutorial prior to this present example and that a
folder named ``seg_sc_t1-t2-t2s-mt`` is available.


Configuration file
------------------

The configuration file used in this tutorial is the one used for
:doc:`../tutorials/one_class_segmentation_2d_unet`. Please open it with a text editor. The parameters that are specific
to this tutorial are:

- ``command``: Action to perform. Here, we want to do some inference using the previously trained model, so we set the
field as follows:

  .. code-block:: xml

     "command": "test"






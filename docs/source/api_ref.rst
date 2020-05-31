API Reference
=============

This document is for developers of ``ivadomed``, it contains the API functions.

:mod:`ivadomed.loader`
----------------------

:mod:`ivadomed.loader.adaptative`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ivadomed.loader.adaptative.Dataframe
.. autoclass:: ivadomed.loader.adaptative.Bids_to_hdf5
.. autoclass:: ivadomed.loader.adaptative.HDF5Dataset
.. automethod:: ivadomed.loader.adaptative.HDF5_to_Bids

:mod:`ivadomed.loader.film`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: ivadomed.loader.film.normalize_metadata
.. autoclass:: ivadomed.loader.film.Kde_model
.. autoclass:: ivadomed.loader.film.clustering_fit
.. autoclass:: ivadomed.loader.film.check_isMRIparam

:mod:`ivadomed.losses`
----------------------

.. autoclass:: ivadomed.losses.MultiClassDiceLoss
.. automethod:: ivadomed.losses.dice_loss
.. autoclass:: ivadomed.losses.FocalLoss
.. autoclass:: ivadomed.losses.FocalDiceLoss
.. autoclass:: ivadomed.losses.GeneralizedDiceLoss

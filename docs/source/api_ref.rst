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
.. automethod:: ivadomed.loader.film.clustering_fit
.. automethod:: ivadomed.loader.film.check_isMRIparam

:mod:`ivadomed.loader.loader`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: ivadomed.loader.loader.load_dataset
.. autoclass:: ivadomed.loader.loader.SegmentationPair
.. autoclass:: ivadomed.loader.loader.MRI2DSegmentationDataset
.. autoclass:: ivadomed.loader.loader.MRI3DSubVolumeSegmentationDataset
.. autoclass:: ivadomed.loader.loader.Bids3DDataset
.. autoclass:: ivadomed.loader.loader.BidsDataset

:mod:`ivadomed.loader.utils`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: ivadomed.loader.utils.split_dataset
.. automethod:: ivadomed.loader.utils.imed_collate
.. automethod:: ivadomed.loader.utils.filter_roi
.. automethod:: ivadomed.loader.utils.orient_img_hwd
.. automethod:: ivadomed.loader.utils.orient_img_ras
.. automethod:: ivadomed.loader.utils.orient_shapes_hwd
.. autoclass:: ivadomed.loader.utils.SampleMetadata
.. autoclass:: ivadomed.loader.utils.BalancedSampler
.. automethod:: ivadomed.loader.utils.clean_metadata
.. automethod:: ivadomed.loader.utils.update_metadata

:mod:`ivadomed.losses`
----------------------

.. autoclass:: ivadomed.losses.MultiClassDiceLoss
.. autoclass:: ivadomed.losses.DiceLoss
.. autoclass:: ivadomed.losses.FocalLoss
.. autoclass:: ivadomed.losses.FocalDiceLoss
.. autoclass:: ivadomed.losses.GeneralizedDiceLoss

:mod:`ivadomed.scripts`
-----------------------

:mod:`ivadomed.scripts.visualize_transforms`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: ivadomed.scripts.visualize_transforms

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

- ``testing_parameters:uncertainty``: Parameters related to the uncertainty estimation. ``"epistemic"`` and
``"aleatoric"`` indicates whether the epistemic and the aleatoric uncertainty are estimated, respectively. Note that
they can be estimated in combination (i.e. both set to ``true``). For more information about these, please read
:doc:`../technical_features#uncertainty-measures`. ``"n_it"`` controls the number of Monte Carlo iterations that are
performed to estimate the uncertainty, please set it to a non-zero positive integer for this tutorial (e.g. ``20``).

  .. code-block:: xml

     "testing_parameters": {
          "uncertainty": {
               "epistemic": true,
               "aleatoric": true,
               "n_it": 20
               }
     }

- ``transformation``: Data-augmentation transformation. If you have selected the aleatoric uncertainty, the data
augmentation that will be performed is the same as the one performed for the training, unless you modify the parameters
in the configuration file (see below). Please make sure ``"dataset_type": ["training"]`` is conserved. Note that only
transformations for which a ``undo_transform`` (i.e. inverse transformation) is available will be performed since these
inverse transformations are required to reconstruct the predicted volume.

  .. code-block:: xml

        "RandomAffine": {
            "degrees": 20,
            "scale": [0.1, 0.1],
            "translate": [0.1, 0.1],
            "applied_to": ["im", "gt"],
            "dataset_type": ["training"]
        },



Selected transformations for the ['testing'] dataset:
	Resample: {'wspace': 0.75, 'hspace': 0.75, 'dspace': 1, 'preprocessing': True}
	CenterCrop: {'size': [128, 128], 'preprocessing': True}
	NumpyToTensor: {}
	NormalizeInstance: {'applied_to': ['im']}


Selected transformations for the ['testing'] dataset:
	Resample: {'wspace': 0.75, 'hspace': 0.75, 'dspace': 1, 'preprocessing': True}
	CenterCrop: {'size': [128, 128], 'preprocessing': True}
	RandomAffine: {'degrees': 5, 'scale': [0.1, 0.1], 'translate': [0.03, 0.03], 'applied_to': ['im', 'gt']}
	ElasticTransform: {'alpha_range': [28.0, 30.0], 'sigma_range': [3.5, 4.5], 'p': 0.1, 'applied_to': ['im', 'gt']}
	NumpyToTensor: {}
	NormalizeInstance: {'applied_to': ['im']}
ElasticTransform transform not included since no undo_transform available for it.




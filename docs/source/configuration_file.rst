Configuration File
==================

All parameters used for loading data, training and predicting are contained
within a single JSON configuration file. This section describes how to set up
this configuration file.

For convenience, here is an generic configuration file:
`config\_config.json <https://raw.githubusercontent.com/ivadomed/ivadomed/master/ivadomed/config/config.json>`__.

Below are other, more specific configuration files:

- `config\_classification.json <https://raw.githubusercontent.com/ivadomed/ivadomed/master/ivadomed/config/config_classification.json>`__: Trains a classification model.

- `config\_sctTesting.json <https://raw.githubusercontent.com/ivadomed/ivadomed/master/ivadomed/config/config_sctTesting.json>`__: Trains a 2D segmentation task with the U-Net architecture.

- `config\_spineGeHemis.json <https://raw.githubusercontent.com/ivadomed/ivadomed/master/ivadomed/config/config_spineGeHemis.json>`__: Trains a segmentation task with the HeMIS-UNet architecture.

- `config\_tumorSeg.json <https://raw.githubusercontent.com/ivadomed/ivadomed/master/ivadomed/config/config_tumorSeg.json>`__: Trains a segmentation task with a 3D U-Net architecture.


General Parameters
------------------


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "command",
        "description": "Run the specified command.",
        "type": "string",
        "options": {"train": "train a model on a training/validation sub-datasets",
                    "test": "evaluate a trained model on a testing sub-dataset",
                    "segment": "segment a entire dataset using a trained model"
        }
    }

.. code-block:: JSON

    {
        "command": "train"
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "gpu_ids",
        "description": "List of IDs of one or more GPUs to use.",
        "type": "list * integer"
    }

.. code-block:: JSON

    {
        "gpu_ids": [1,2,3]
    }

.. note::
    Currently only ``ivadomed_automate_training`` supports the use of more than one GPU.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "log_directory",
        "description": "Folder name that will contain the output files (e.g., trained model,
            predictions, results).",
        "type": "string"
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "model_name",
        "description": "Folder name containing the trained model (ONNX format) and its configuration
            file, located within ``log_directory/``",
        "type": "string"
    }

Example:

.. code-block:: sh

    "log_directory/seg_gm_t2star/seg_gm_t2star.onnx"
    "log_directory/seg_gm_t2star/seg_gm_t2star.json"

When possible, the folder name will follow the following convention:
``task_(animal)_region_(contrast)`` with

.. code-block:: sh

   task = {seg, label, find}
   animal = {human, dog, cat, rat, mouse, ...}
   region = {sc, gm, csf, brainstem, ...}
   contrast = {t1, t2, t2star, dwi, ...}


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "debugging",
        "description": "Extended verbosity and intermediate outputs.",
        "type": "boolean"
    }

Loader parameters
-----------------

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "bids_path",
        "description": "Path of the BIDS folder.",
        "type": "string"
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "bids_config",
        "description": "(Optional). Path of the custom BIDS configuration file for
            BIDS non-compliant modalities",
        "type": "string"
    }

Example:

.. code-block:: JSON

    {
        "bids_config": "ivadomed/config/config_bids.json"
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "subject_selection",
        "description": "Used to specify a custom subject selection from a dataset.",
        "type": "dict",
        "options": {
            "n": {
                "description": "List containing the number subjects of each metadata.",
                "type": "list"
            },
            "metadata": {
                "$$description": [
                    "List of metadata used to select the subjects. Each metadata should be the name\n",
                    "of a column from the participants.tsv file."
                ],
                "type": "list"
            },
            "value": {
                "description": "List of metadata values of the subject to be selected.",
                "type": "list"
            }
        }
    }


Example:

.. code-block:: JSON

    {
        "subject_selection": {
            "n": [5, 10],
            "metadata": ["disease", "disease"],
            "value": ["healthy", "ms"]
        }
    }

In this example, a subdataset composed of 5 healthy subjects and 10 ms subjects will be selected
for training/testing.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "target_suffix",
        "description": "Suffix list of the derivative file containing the ground-truth of interest.",
        "type": "list * string"
    }

Example:

.. code-block:: JSON

    {
        "target_suffix": ["_seg-manual", "_lesion-manual"]
    }

The length of this list controls the number of output channels of the model (i.e.
``out_channel``). If the list has a length greater than 1, then a
multi-class model will be trained. If a list of list(s) is input for a
training, (e.g. [[``"_seg-manual-rater1"``, ``"_seg-manual-rater2"``],
[``"_lesion-manual-rater1"``, ``"_lesion-manual-rater2"``]), then each
sublist is associated with one class but contains the annotations from
different experts: at each training iteration, one of these annotations
will be randomly chosen.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "extensions",
        "description": "Used to specify a list of file extensions to be selected for
            training/testing.",
        "type": "list * string"
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "contrasts",
        "type": "dict",
        "options": {
            "train_validation": {
                "type": "list, string",
                "$$description": [
                    "List of image contrasts (e.g. ``T1w``, ``T2w``) loaded for the training and\n",
                    "validation. If ``multichannel`` is ``true``, this list represents the different\n",
                    "channels of the input tensors (i.e. its length equals model's ``in_channel``).\n",
                    "Otherwise, the contrasts are mixed and the model has only one input channel\n",
                    "(i.e. model's ``in_channel=1``)"
                ]
            },
            "test": {
                "type": "list, string",
                "$$description": [
                    "List of image contrasts (e.g. ``T1w``, ``T2w``) loaded in the testing dataset.\n",
                    "Same comment as for ``train_validation`` regarding ``multichannel``."
                ]
            },
            "balance": {
                "type": "dict",
                "$$description": [
                    "Enables to weight the importance of specific channels (or contrasts) in the\n",
                    "dataset: e.g. ``{'T1w': 0.1}`` means that only 10% of the available ``T1w``\n",
                    "images will be included into the training/validation/test set. Please set\n",
                    "``multichannel`` to ``false`` if you are using this parameter."
                ]
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "multichannel",
        "description": "Indicated if more than a contrast (e.g. ``T1w`` and ``T2w``) is
            used by the model.",
        "type": "boolean"
    }

See details in both ``train_validation`` and ``test`` for the contrasts that are input.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "slice_axis",
        "description": "Sets the slice orientation for on which the model will be used.",
        "type": "string",
        "options": {"sagittal": "plane dividing body into left/right",
                    "coronal": "plane dividing body into front/back",
                    "axial": "plane dividing body into top/bottom"
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "slice_filter",
        "description": "Discard a slice from the dataset if it meets a condition, see
            below.",
        "type": "dict",
        "options": {
            "filter_empty_input": {
                "type": "boolean",
                "description": "Discard slices where all voxel
                   intensities are zeros."
            },
            "filter_empty_mask": {
                "type": "boolean",
                "description": "Discard slices where all voxel labels are zeros."
            },
            "filter_absent_class": {
                "type": "boolean",
                "$$description": [
                    "Discard slices where all voxel labels are zero for one or more classes\n",
                    "(this is most relevant for multi-class models that need GT for all classes at train time)."
                ]
            },
            "filter_classification": {
                "type": "boolean",
                "$$description": [
                    "Discard slices where all images fail a custom classifier filter. If used,\n",
                    "``classifier_path`` must also be specified, pointing to a saved PyTorch classifier."
                ]
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "roi",
        "description": "Parameters for the region of interest",
        "type": "dict",
        "options": {
            "suffix": {
                "type": "string",
                "$$description": [
                    "Suffix of the derivative file containing the ROI used to crop\n",
                    "(e.g. ``_seg-manual``) with ``ROICrop`` as transform. Please use ``null`` if",
                    "you do not want to use an ROI to crop."
                ]
            },
            "slice_filter_roi": {
                "type": "int",
                "$$description": [
                    "If the ROI mask contains less than ``slice_filter_roi`` non-zero voxels\n",
                    "the slice will be discarded from the dataset. This feature helps with\n",
                    "noisy labels, e.g., if a slice contains only 2-3 labeled voxels, we do\n",
                    "not want to use these labels to crop the image. This parameter is only\n",
                    "considered when using ``ROICrop``."
                ]
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "soft_gt",
        "$$description": [
            "Indicates if a soft mask will be used as ground-truth to train\n",
            "and / or evaluate a model. In particular, the masks are not binarized\n",
            "after interpolations implied by preprocessing or data-augmentation operations."
        ],
        "type": "boolean"
    }


Split Dataset
-------------

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "fname_split",
        "$$description": [
            "File name of the log (`joblib <https://joblib.readthedocs.io/en/latest/>`__)\n",
            "that contains the list of training/validation/testing subjects. This file can later\n",
            "be used to re-train a model using the same data splitting scheme. If ``null``,\n",
            "a new splitting scheme is performed."
        ],
        "type": "string"
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "random_seed",
        "$$description": [
            "Seed used by the random number generator to split the dataset between\n",
            "training/validation/testing. The use of the same seed ensures the same split between\n",
            "the sub-datasets, which is useful for reproducibility."
        ],
        "type": "int"
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "method",
        "$$description": [
            "Seed used by the random number generator to split the dataset between\n",
            "training/validation/testing. The use of the same seed ensures the same split between\n",
            "the sub-datasets, which is useful for reproducibility."
        ],
        "type": "string",
        "options": {
            "per_patient": {
                "$$description": [
                    "all subjects are shuffled, then split between train/validation/test\n",
                    "according to ``train_fraction`` and ``test_fraction``, regardless of their institution"
                ]
            },
            "per_center": {
                "$$description": [
                    "all subjects are split so as not to mix institutions between the\n",
                    "train/validation/test sets according to ``train_fraction`` and ``center_test``.\n",
                    "The latter option enables the user to ensure the model is working across domains (institutions)."
                ]
            }
        }
    }

.. note::
    The institution information is contained within the ``institution_id`` column in the
    ``participants.tsv`` file.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "balance",
        "$$description": [
            "Metadata contained in ``participants.tsv`` file with categorical values. Each category\n",
            "will be evenly distributed in the training, validation and testing datasets."
        ],
        "type": "string",
        "required": "false"
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "train_fraction",
        "description": "Fraction of the dataset used as training set.",
        "type": "float",
        "range": "[0, 1]"
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "test_fraction",
        "$$description": [
            "Fraction of the dataset used as testing set. This parameter is only used if the\n",
            "``method`` is ``per_patient``"
        ],
        "type": "float",
        "range": "[0, 1]"
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "center_test",
        "$$description": [
            "Each string corresponds to an institution/center to only include in the testing\n",
            "dataset (not validation). This parameter is only used if the ``method`` is ``per_center``\n",
            "If used, the file ``bids_dataset/participants.tsv`` needs to contain a column\n",
            "``institution_id``, which associates a subject with an institution/center."
        ],
        "type": "list, string"
    }



Training Parameters
-------------------


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "batch_size",
        "type": "int",
        "range": "(0, inf)"
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "loss",
        "$$description": [
            "Metadata for the loss function. Other parameters that could be needed in the\n",
            "Loss function definition: see attributes of the Loss function of interest\n",
            "(e.g. ``'gamma': 0.5`` for ``FocalLoss``)."
        ],
        "type": "dict",
        "options": {
            "name": {
                "type": "string",
                "description": "Name of the loss function class. See :mod:`ivadomed.losses`"
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "training_time",
        "$$description": [
            "Metadata for the loss function. Other parameters that could be needed in the\n",
            "Loss function definition: see attributes of the Loss function of interest\n",
            "(e.g. ``'gamma': 0.5`` for ``FocalLoss``)."
        ],
        "type": "dict",
        "options": {
            "num_epochs": {
                "type": "int",
                "range": "(0, inf)"
            },
            "early_stopping_epsilon": {
                "type": "float",
                "$$description": [
                    "If the validation loss difference during one epoch\n",
                    "(i.e. ``abs(validation_loss[n] - validation_loss[n-1]`` where n is the current epoch)\n",
                    "is inferior to this epsilon for ``early_stopping_patience`` consecutive epochs,\n",
                    "then training stops."
                ]
            },
            "early_stopping_patience": {
                "type": "int",
                "range": "(0, inf)",
                "$$description": [
                    "Number of epochs after which the training is stopped if the validation loss\n",
                    "improvement is smaller than ``early_stopping_epsilon``."
                ]
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "scheduler",
        "type": "dict",
        "options": {
            "initial_lr": {
                "type": "float",
                "description": "Initial learning rate."
            },
            "scheduler_lr": {
                "type": "dict",
                "options": {
                    "name": {
                        "type": "string",
                        "$$description": [
                            "One of ``CosineAnnealingLR``, ``CosineAnnealingWarmRestarts``\n",
                            "and ``CyclicLR``. Please find documentation `here <https://pytorch.org/docs/stable/optim.html>`__.\n",

                        ]
                    }
                },
                "description": "Other parameters depend on the scheduler of interest"
            }
        }
    }

Example:

.. code-block:: JSON

    {
        "scheduler":
            "initial_lr": 0.001,
            "scheduler_lr": {
                "name": "CosineAnnealingLR",
                "max_lr": 1e-2,
                "base_lr": 1e-5
            }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "balance_samples",
        "description": "Balance labels in both the training and the validation datasets.",
        "type": "dict",
        "options": {
          "applied": {
              "type": "boolean",
              "description": "Indicates whether to use a balanced sampler or not."
          },
          "type": {
              "type": "string",
              "$$description": [
                "Indicates which metadata to use to balance the sampler.\n",
                "Choices: ``gt`` or  the name of a column from the ``participants.tsv`` file\n",
                "(i.e. subject-based metadata)"
              ]
          }
        }
     }

Example:

.. code-block:: JSON

    "balance_samples": {
        "applied": false,
        "type": "gt"
    }



.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "mixup_alpha",
        "description": "Alpha parameter of the Beta distribution, see `original paper on
        the Mixup technique <https://arxiv.org/abs/1710.09412>`__.",
        "type": "float"
    }


.. jsonschema::

   {
       "$schema": "http://json-schema.org/draft-04/schema#",
       "title": "transfer_learning",
       "type": "dict",
       "options": {
           "retrain_model": {
               "type": "string",
               "$$description": [
                   "Filename of the pretrained model (``path/to/pretrained-model``). If ``null``,\n",
                   "no transfer learning is performed and the network is trained from scratch."
               ]
           },
           "retrain_fraction": {
               "type": "float",
               "range": "[0, 1]",
               "$$description": [
                   "Controls the fraction of the pre-trained model that will be fine-tuned. For\n",
                   "instance, if set to 0.5, the second half of the model will be fine-tuned while\n",
                   "the first layers will be frozen."
               ]
           },
           "reset": {
               "type": "boolean",
               "description": "If true, the weights of the layers that are not frozen
                  are reset. If false, they are kept as loaded."
           }
       }
   }

Architecture
------------

Architectures for both segmentation and classification are available and
described in the :ref:`architectures` section. If the selected architecture is listed in the
`loader <https://github.com/ivadomed/ivadomed/blob/lr/fixing_documentation/ivadomed/loader/loader.py>`__ file, a
classification (not segmentation) task is run. In the case of a
classification task, the ground truth will correspond to a single label
value extracted from ``target``, instead being an array (the latter
being used for the segmentation task).


.. jsonschema::

   {
       "$schema": "http://json-schema.org/draft-04/schema#",
       "title": "default_model",
       "required": "true",
       "type": "dict",
       "$$description": [
           "Define the default model (``Unet``) and mandatory parameters that are common to all\n",
           "available :ref:`architectures`. For custom architectures (see below), the default\n",
           "parameters are merged with the parameters that are specific to the tailored architecture."
       ],
       "options": {
           "name": {
               "type": "string",
               "description": "Default: ``Unet``"
           },
           "dropout_rate": {
               "type": "float"
           },
           "batch_norm_momentum": {
               "type": "float"
           },
           "depth": {
               "type": "int",
               "range": "(0, inf)",
               "description": "Number of down-sampling operations."
           },
           "relu": {
               "type": "boolean",
               "required": "false",
               "$$description": [
                   "Sets final activation to normalized ReLU (relu between 0 and 1), instead of\n",
                   "sigmoid. Only available when `is_2D=True`."
               ]
           },
           "is_dim": {
               "type": "boolean",
               "$$description": [
                   "Indicates dimensionality of model (2D or 3D). If ``is_dim`` is ``False``, then parameters\n",
                   "``length_3D`` and ``stride_3D`` for 3D loader need to be specified (see :ref:`Modified3DUNet <Modified3DUNet>`)."
               ]
           }
       }
   }


Example:

.. code-block:: JSON

    {
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.4,
            "batch_norm_momentum": 0.1
        }
    }



.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "FiLMedUnet",
        "type": "dict",
        "required": "false",
        "options": {
            "applied": {
                "type": "boolean",
                "description": "Set to ``true`` to use this model."
            },
            "metadata": {
                "type": "string",
                "options": {
                    "mri_params": {
                        "$$description": [
                            "Vectors of ``[FlipAngle, EchoTime, RepetitionTime, Manufacturer]``\n",
                            "(defined in the json of each image) are input to the FiLM generator."
                        ]
                    },
                    "contrast": "Image contrasts (according to ``config/contrast_dct.json``) are input to the FiLM generator."
               },
               "$$description": [
                   "Choice between ``mri_params``, ``contrasts`` (i.e. image-based metadata) or the\n",
                   "name of a column from the participants.tsv file (i.e. subject-based metadata)."
               ]
           }
       }
   }


.. jsonschema::


  	{
		"$schema": "http://json-schema.org/draft-04/schema#",
      	"title": "HeMISUnet",
      	"type": "dict",
      	"required": "false",
      	"options": {
			"applied": {
				"type": "boolean",
              	"description": "Set to ``true`` to use this model."
			},
          	"missing_probability": {
				"type": "float",
                "range": "[0, 1]",
                "$$description": [
                    "Initial probability of missing image contrasts as model's input\n",
                    "(e.g. 0.25 results in a quarter of the image contrasts, i.e. channels, that\n",
                    "will not be sent to the model for training)."
                ]
            },
            "missing_probability_growth": {
                "type": "float",
                "$$description": [
                    "Controls missing probability growth at each epoch: at each epoch, the\n",
                    "``missing_probability`` is modified with the exponent ``missing_probability_growth``."
                ]
            }
         }
      }

.. _Modified3DUNet:

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "Modified3DUNet",
        "type": "dict",
        "required": "false",
        "options": {
            "length_3D": {
                "type": "(int, int, int)",
                "description": "Size of the 3D patches used as model's input tensors."
            },
            "stride_3D": {
                "type": "[int, int, int]",
                "$$description": [
                    "Voxels' shift over the input matrix to create patches. Ex: Stride of [1, 2, 3]\n",
                    "will cause a patch translation of 1 voxel in the 1st dimension, 2 voxels in\n",
                    "the 2nd dimension and 3 voxels in the 3rd dimension at every iteration until\n",
                    "the whole input matrix is covered."
                ]
            },
            "attention_unet": {
                "type": "boolean",
                "description": "Use attention gates in the Unet's decoder.",
                "required": "false"
            },
            "n_filters": {
                "type": "int",
                "$$description": [
                    "Number of filters in the first convolution of the UNet.\n",
                    "This number of filters will be doubled at each convolution."
                ],
                "required": "false"
            }
       }
   }

Example:

.. code-block:: JSON

    {
        "Modified3DUNet": {
            "applied": false,
            "length_3D": [128, 128, 16],
            "stride_3D": [128, 128, 16],
            "attention": false,
            "n_filters": 8
        }
    }


Cascaded Architecture Features
------------------------------

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "object_detection_params",
        "type": "dict",
        "required": "false",
        "options": {
            "object_detection_path": {
                "type": "string",
                "$$description": [
                    "Path to object detection model and the configuration file. The folder,\n",
                    "configuration file, and model need to have the same name\n",
                    "(e.g. ``findcord_tumor/``, ``findcord_tumor/findcord_tumor.json``, and\n",
                    "``findcord_tumor/findcord_tumor.onnx``, respectively). The model's prediction\n",
                    "will be used to generate bounding boxes."
                ]
            },
            "safety_factor": {
                "type": "[int, int, int]",
                "$$description": [
                    "List of length 3 containing the factors to multiply each dimension of the\n",
                    "bounding box. Ex: If the original bounding box has a size of 10x20x30 with\n",
                    "a safety factor of [1.5, 1.5, 1.5], the final dimensions of the bounding box\n",
                    "will be 15x30x45 with an unchanged center."
                ]
            }
       }
   }


Transformations
---------------

Transformations applied during data augmentation. Transformations are
sorted in the order they are applied to the image samples. For each
transformation, the following parameters are customizable: -
``applied_to``: list betweem ``"im", "gt", "roi"``. If not specified,
then the transformation is applied to all loaded samples. Otherwise,
only applied to the specified types: eg ``["gt"]`` implies that this
transformation is only applied to the ground-truth data. -
``dataset_type``: list between ``"training", "validation", "testing"``.
If not specified, then the transformation is applied to the three
sub-datasets. Otherwise, only applied to the specified subdatasets: eg
``["testing"]`` implies that this transformation is only applied to the
testing sub-dataset.

Available Transformations:
^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``NumpyToTensor``
-  ``CenterCrop2D`` (parameters: ``size``)
-  ``ROICrop2D`` (parameters: ``size``)
-  ``NormalizeInstance``
-  ``RandomAffine`` (parameters: ``degrees`` (Positive integer),
   ``translate`` (List of floats between 0. and 1.), ``scale`` (List of
   floats between 0. and 1.))
-  ``RandomShiftIntensity`` (parameters: ``shift_range``)
-  ``ElasticTransform`` (parameters: ``alpha_range``, ``sigma_range``,
   ``p``)
-  ``Resample`` (parameters: ``wspace``, ``hspace``, ``dspace``)
-  ``AdditionGaussianNoise`` (parameters: ``mean``, ``std``)
-  ``DilateGT`` (parameters: ``dilation_factor``) Float. Controls the
   number of iterations of ground-truth dilation depending on the size
   of each individual lesion, data augmentation of the training set. Use
   ``0`` to disable.
-  ``HistogramClipping`` (parameters: ``min_percentile``,
   ``max_percentile``)
-  ``Clahe`` (parameters: ``clip_limit``, ``kernel_size``)
-  ``RandomReverse``

.. _Uncertainty:

Uncertainty
-----------

Uncertainty computation is performed if ``n_it>0`` and at least
``epistemic`` or ``aleatoric`` is ``true``. Note: both ``epistemic`` and
``aleatoric`` can be ``true``.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "epistemic",
        "type": "boolean",
        "description": "Model-based uncertainty with `Monte Carlo Dropout <https://arxiv.org/abs/1506.02142>`__."
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "aleatoric",
        "type": "boolean",
        "description": "Image-based uncertainty with `test-time augmentation <https://doi.org/10.1016/j.neucom.2019.01.103>`__."
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "n_it",
        "type": "int",
        "description": "Number of Monte Carlo iterations. Set to 0 for no uncertainty computation."
    }


Postprocessing
--------------

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "binarize_prediction",
        "type": "dict",
        "options": {
            "thr": {
                "type": "float",
                "range": "[0, 1]",
                "$$description": [
                    "Threshold. To use soft predictions (i.e. no binarisation, float between 0 and 1)\n",
                    "for metric computation, indicate -1."
                ]
            }
        },
        "$$description": [
            "Binarizes predictions according to the given threshold ``thr``. Predictions below the\n",
            "threshold become 0, and predictions above or equal to threshold become 1."
        ]
    }

Example:

.. code-block:: JSON

    {
        "postprocessing": {
            "binarize_prediction": {
                "thr": 0.1
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "binarize_maxpooling",
        "type": "dict",
        "$$description": [
            "Binarize by setting to 1 the voxel having the maximum prediction across all classes.\n",
            "Useful for multiclass models. No parameters required (i.e., {})."
        ]
    }

Example:

.. code-block:: JSON

    {
        "postprocessing": {
            "binarize_maxpooling": {}
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "fill_holes",
        "type": "dict",
        "description": "Fill holes in the predictions. No parameters required (i.e., {})."
    }

Example:

.. code-block:: JSON

    {
        "postprocessing": {
            "fill_holes": {}
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "keep_largest",
        "type": "dict",
        "$$description": [
            "Keeps only the largest connected object in prediction. Only nearest neighbors are\n",
            "connected to the center, diagonally-connected elements are not considered neighbors.\n",
            "No parameters required (i.e., {})"
        ]
    }

Example:

.. code-block:: JSON

    {
        "postprocessing": {
            "keep_largest": {}
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "remove_noise",
        "type": "dict",
        "options": {
            "thr": {
                "type": "float",
                "range": "[0, 1]",
                "description": "Threshold. Threshold set to ``-1`` will not apply this postprocessing step."
            }
        },
        "description": "Sets to zero prediction values strictly below the given threshold ``thr``."
    }

Example:

.. code-block:: JSON

    {
        "postprocessing": {
            "remove_noise": {
                "thr": 0.1
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "remove_small",
        "type": "dict",
        "$$description": [
            "Remove small objects from the prediction. An object is defined as a group of connected\n",
            "voxels. Only nearest neighbors are connected to the center, diagonally-connected\n",
            "elements are not considered neighbors."
        ],
        "options": {
            "thr": {
                "type": "int or list",
                "$$description": [
                    "Minimal object size. If a list of thresholds is chosen, the length should\n",
                    "match the number of predicted classes."
                ]
            },
            "unit": {
                "type": "string",
                "$$description": [
                    "Either `vox` for voxels or `mm3`. Indicates the unit used to define the\n",
                    "minimal object size."
                ]
            }
        }
    }

Example:

.. code-block:: JSON

    {
        "postprocessing": {
            "remove_small": {
                "unit": "vox",
                "thr": 3
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "threshold_uncertainty",
        "type": "dict",
        "$$description": [
            "Removes the most uncertain predictions (set to 0) according to a threshold ``thr``\n",
            "using the uncertainty file with the suffix ``suffix``. To apply this method,\n",
            "uncertainty needs to be evaluated on the predictions with the :ref:`uncertainty <Uncertainty>` parameter."
        ],
        "options": {
            "thr": {
                "type": "float",
                "range": "[0, 1]",
                "$$description": [
                    "Threshold. Threshold set to ``-1`` will not apply this postprocessing step."
                ]
            },
            "suffix": {
                "type": "string",
                "$$description": [
                    "Indicates the suffix of an uncertainty file. Choices: ``_unc-vox.nii.gz`` for\n",
                    "voxel-wise uncertainty, ``_unc-avgUnc.nii.gz`` for structure-wise uncertainty\n",
                    "derived from mean value of ``_unc-vox.nii.gz`` within a given connected object,\n",
                    "``_unc-cv.nii.gz`` for structure-wise uncertainty derived from coefficient of\n",
                    "variation, ``_unc-iou.nii.gz`` for structure-wise measure of uncertainty\n",
                    "derived from the Intersection-over-Union of the predictions, or ``_soft.nii.gz``\n",
                    "to threshold on the average of Monte Carlo iterations."
                ]
            }
        }
    }

Example:

.. code-block:: JSON

    {
        "postprocessing": {
            "threshold_uncertainty": {
                "thr": -1,
                "suffix": "_unc-vox.nii.gz"
            }
        }
    }


Evaluation Parameters
---------------------
Dict. Parameters to get object detection metrics (true positive and false detection rates), and this, for defined
object sizes.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "targetSize",
        "type": "dict",
        "options": {
            "thr": {
                "type": "list, int",
                "$$description": [
                    "These values will create several consecutive target size bins. For instance\n",
                    "with a list of two values, we will have three target size bins: minimal size\n",
                    "to first list element, first list element to second list element, and second\n",
                    "list element to infinity."
                ]
            },
            "unit": {
                "type": "string",
                "$$description": [
                    "Either `vox` for voxels or `mm3`. Indicates the unit used to define the\n",
                    "target object sizes."
                ]
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "overlap",
        "type": "dict",
        "options": {
            "thr": {
                "type": "int",
                "$$description": [
                    "Minimal object size overlapping to be considered a TP, FP, or FN."
                ]
            },
            "unit": {
                "type": "string",
                "$$description": [
                    "Either `vox` for voxels or `mm3`. Indicates the unit used to define the\n",
                    "overlap."
                ]
            }
        }
    }



Examples
--------

Examples of configuration files: `config\_config.json <ivadomed/config/config.json>`__.

In particular:

- `config\_classification.json <ivadomed/config/config_classification.json>`__. Is dedicated to classification task.

- `config\_sctTesting.json <ivadomed/config/config_sctTesting.json>`__. Is a user case of 2D segmentation using a U-Net model.

- `config\_spineGeHemis.json <ivadomed/config/config_spineGeHemis.json>`__. Shows how to use the HeMIS-UNet.

- `config\_tumorSeg.json <ivadomed/config/config_tumorSeg.json>`__. Runs a 3D segmentation using a 3D UNet.

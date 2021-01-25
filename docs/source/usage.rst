Usage
=====

Command line tools
------------------

New model can be generated using the command-line tool from the
terminal:

::

    ivadomed [command] -c path/to/config_file.json --path-data path/to/bids/data --path-output path/to/output/directory

where ``[command]`` represents the following choice of flag: ``--train``: train a model on a training/validation
sub-datasets, ``--test``: evaluate a trained model on a testing sub-dataset, ``--segment``:
segment a entire dataset using a trained model. Note that you may only specify one command flag at a time.

where ``config_file.json`` is a configuration file, which parameters are
described in the :ref:`configuration_file:Configuration File`.

where ``path/to/bids/data`` is the location of the dataset. As discussed in `Data <../data.html>`__, the dataset 
should conform to the BIDS standard. Modify the path so it points to the location of the downloaded dataset.

where ``path/to/output/directory`` is the folder name that will contain the output files (e.g., trained model, predictions, results)

Please see section ``TUTORIALS`` to run this command on an example dataset.

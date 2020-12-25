Data
====

To facilitate the organization of data, ``ivadomed`` requires the data to be
organized according to the `Brain Imaging Data Structure (BIDS) <http://bids.neuroimaging.io/>`__ convention.
An example of this organization is shown below:

.. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/data/1920px-BIDS_Logo.png
    :alt: BIDS_Logo

::

    dataset/
    └── dataset_description.json
    └── participants.tsv
    └── sub-01
    └── sub-02
    └── sub-03
        └── anat
            └── sub-03_T1w.nii.gz  <-- MRI image in NIfTI format
            └── sub-03_T1w.json  <---- Metadata including image parameters, MRI vendor, etc.
            └── sub-03_T2w.nii.gz
            └── sub-03_T2w.json
    └── derivatives
        └── labels
            └── sub-03
                └── anat
                    └── sub-03_seg-tumor-manual.nii.gz  <-- Manually-corrected segmentation
                    └── sub-03_seg-tumor-manual.json  <---- Metadata including author who performed the labeling and date

.. note:: ``participants.tsv`` should, at least, include a column ``participant_id``, which is used when loading the dataset.

.. note:: For an exhaustive list of derivatives used in ``ivadomed``, please see our `wiki <https://github.com/ivadomed/ivadomed/wiki/repositories#derivatives>_`

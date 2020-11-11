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
        └── anat
            └── sub-siteX01_T1w_reg.nii.gz
            └── sub-siteX01_T1w_reg.json
            └── sub-siteX01_T2w_reg.nii.gz
            └── sub-siteX01_T2w_reg.json
            └── sub-siteX01_acq-MTon_MTS_reg.nii.gz
            └── sub-siteX01_acq-MTon_MTS_reg.json
            └── sub-siteX01_acq-MToff_MTS_reg.nii.gz
            └── sub-siteX01_acq-MToff_MTS_reg.json
            └── sub-siteX01_acq-T1w_MTS.nii.gz
            └── sub-siteX01_acq-T1w_MTS.json
            └── sub-siteX01_T2star_reg.nii.gz
            └── sub-siteX01_T2star_reg.json
    └── derivatives
        └── labels
            └── sub-siteX01
                └── anat
                    └── sub-siteX01_T1w_seg.nii.gz

.. note:: ``participants.tsv`` should, at least, include a column ``participant_id``, which is used when loading the dataset.

Data
====

Without data, nothing can be done. To get you started, we recommend you
download the `Spinal Cord MRI Public
Database <https://openneuro.org/datasets/ds001919>`__. This dataset is
composed of 248+ subjects from different imaging centers and includes
original images in NIfTI format as well as manual segmentations and
labels. The data are organized according to the
`BIDS <http://bids.neuroimaging.io/>`__ convention, to be fully
compatible with ``ivadomed`` loader:

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

.. warning:: ``TODO: Update openneuro site to include derivatives``

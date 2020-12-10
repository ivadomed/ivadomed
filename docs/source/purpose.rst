.. |yes| raw:: html

   <style> .line {text-align:centers;} </style>
   <p style="color:green" align="center">	&#10004;</p>

.. |no| raw:: html

   <style> .line {text-align:centers;} </style>
   <p style="color:red" align="center">	&#10007;</p>

.. |cent| raw:: html

  <style> .line {text-align:center;} </style>


Purpose
=======

The purpose of the ``ivadomed`` project is to:

* Provide researchers with an open-source framework for training deep learning models for applications in medical imaging;

* Provide ready-to-use :doc:`pretrained_models` trained on multi-center data.

Comparison with other projects
------------------------------

We acknowledge the existence of projects with similar purposes. The table below compares some features across some
of the existing projects. This table was mostly based on the existing documentation for each project. We
understand that the field is rapidly evolving, and that this table might reflect the reality. If you notice
inconsistencies, please let us know by `opening an issue <https://github.com/ivadomed/ivadomed/issues>`_.

..
  If you wish to modify the csv tbale please modify https://docs.google.com/spreadsheets/d/1_MydnHnlOAuYzJ9QBCvPC9Jq2xUmPWI-XttTfcdtW2Y/edit#gid=0

.. csv-table::
   :file: comparison_other_projects_table.csv

(1): `"BIDS" stands for the Brain Imaging Data Structure <https://bids.neuroimaging.io/>`_, which is a convention initiated by the neuroimaging community to organize datasets (filenames, metadata, etc.). This facilitates the sharing of datasets and minimizes the burden of organizing datasets for training.

(2): Class: Classification | Seg: Segmentation | Detect: Detection | Gen: Generation | Clust: Clustering | Reg: Registration




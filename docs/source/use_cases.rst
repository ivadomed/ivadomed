Use cases
=========

Use case #1 - Spinal Cord Toolbox:
----------------------------------

`Spinal cord toolbox <http://spinalcordtoolbox.com/>`__ (SCT) is an open-source analysis software package for processing MRI data of the spinal cord `[De Leener et al. 2017] <https://doi.org/10.1016/j.neuroimage.2016.10.009>`__. `ivadomed` is SCT's backbone for the automated segmentation of the spinal cord, gray matter, tumors, and multiple sclerosis lesions, as well as for the labeling of intervertebral discs.

Use case 2 - Creation of anatomical template:
---------------------------------------------

`ivadomed` was used in the generation of several high-resolution anatomical MRI templates `[Calabrese et al. 2018 <https://doi.org/10.1038/s41598-018-24304-3>`__, `Gros et al. 2020] <https://github.com/sct-pipeline/exvivo-template>`__. To make anatomical templates, it is sometimes necessary to segment anatomical regions, such as the spinal cord white matter. When dealing with high resolution data, there may be several thousand 2D slices to segment, stressing the need for a fully-automated and robust solution. In these studies, only a handful of slices were manually-segmented and used to train a specific model. The model then predicted reliably and with high accuracy (Dice score > 90%) the delineation of anatomical structures for the thousands of remaining unlabeled slices.


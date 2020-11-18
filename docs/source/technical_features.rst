Technical features
==================

Physics-informed network
------------------------

CNNs can be modulated, at each layer, using the `Feature-wise Linear
Modulation (FiLM) <https://arxiv.org/abs/1709.07871>`__ technique.
FiLM permits to add priors during training/inference based on the
imaging physics (e.g. acquisition parameters), thereby improving the
performance of the output segmentations.

.. figure:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/technical_features/film_figure.png
   :alt: Figure FiLM

   Figure FiLM

.. _Uncertainty-measures:

Uncertainty measures
--------------------

At inference time, uncertainty can be estimated via two ways: -
model-based uncertainty (epistemic) based on `Monte Carlo
Dropout <https://arxiv.org/abs/1506.02142>`__. - image-based uncertainty
(aleatoric) `based on test-time
augmentation <https://doi.org/10.1016/j.neucom.2019.01.103>`__.

From the Monte Carlo samples, different measures of uncertainty can be
derived: - voxel-wise entropy - structure-wise intersection over union -
structure-wise coefficient of variation - structure-wise averaged
voxel-wise uncertainty within the structure

These measures can be used to perform some
`post-processing <https://arxiv.org/abs/1808.01200>`__ based on the
uncertainty measures.

.. figure:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/technical_features/uncertainty_measures.png
   :alt: Figure Uncertainty

   Figure Uncertainty

Two-step training scheme with class sampling
--------------------------------------------

Class sampling, coupled with a transfer learning strategy, can mitigate
class imbalance issues, while addressing the limitations of classical
under-sampling (risk of loss of information) or over-sampling (risk of
overfitting) approaches.

During a first training step, the CNN is trained on an equivalent
proportion of positive and negative samples, negative samples being
under-weighted dynamically at each epoch. During the second step, the
CNN is fine-tuned on the realistic (i.e. class-imbalanced) dataset.

Mixup
-----

`Mixup <https://arxiv.org/abs/1710.09412>`__ is a data augmentation
technique, wherein training is performed on samples that are generated
by combining two random samples from the training set and from the
associated labels. The motivation is to regularize the network while
extending the training distribution.

.. figure:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/technical_features/mixup.png
   :alt: Figure mixup

   Figure mixup

Data augmentation on lesion labels
----------------------------------

This data augmentation is motivated by the large inter-rater variability
that is common in medical image segmentation tasks. Typically, raters
disagree on the boundaries of pathologies (e.g., tumors, lesions). A
soft mask is constructed by morphological dilation of the binary
segmentation (i.e. mask provided by expert), where expert-labeled voxels
have one as value while the augmented voxels are assigned a soft value
which depends on the distance to the core of the lesion. Thus, the prior
knowledge about the subjective lesion borders is then leveraged to the
network.

.. figure:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/technical_features/dilate-gt.png
   :alt: Figure Data Augmentation on lesion ground truths

   Figure Data Augmentation on lesion ground truths

Network architectures
---------------------

-  `UNet <https://arxiv.org/abs/1505.04597>`__, with control of the
   network depth.
-  HeMIS-UNet: integrates the
   `HeMIS <https://arxiv.org/abs/1607.05194>`__ strategy to deal with
   missing modalities within a UNet training scheme.
-  FiLMed-UNet, based on `FiLM <https://arxiv.org/abs/1709.07871>`__
   strategy adapted to the `segmentation
   task <#physic-informed-network>`__.
- Countception: modified implementation of `Countception <https://arxiv.org/abs/1703.08710>`__ for keypoints detection.

Loss functions
--------------

-  `Dice Loss <https://arxiv.org/abs/1606.04797>`__. Also adapted for
   multi-label segmentation tasks, by averaging the loss for each class.
-  `Focal Loss <https://arxiv.org/abs/1708.02002>`__.
-  Focal-Dice Loss: Linear combination of the Focal and Dice losses.
-  `Generalized Dice Loss <https://arxiv.org/abs/1707.03237>`__. An
   additional feature compared to the published reference, is that the
   background volume can be weighted by the inverse of its area, which
   could be of interest in high class imbalance scenarios.
-  `Adaptive wing loss <https://arxiv.org/abs/1904.07399>`__. Loss function used to detect key points with Gaussian representation of the target.
-  Loss Combination: Linear combination of any other implemented losses. 

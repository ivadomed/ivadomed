***************
Neural Networks
***************

If you're new to machine learning/neural networks, ``Coursera`` offers a great introduction:
`Coursera Machine Learning Course <https://www.coursera.org/learn/machine-learning>`_.

We won't exhaustively cover all the details of neural networks here, but will give a brief
overview of some of specific neural networks and features we use in ``ivadomed``.

Loss Function
=============

In any given neural network, a loss function is computed for each iteration. The loss function
computes how close the predictions are to the target values.

As a simple example, let's say I wanted to build a neural network that classifies images into
the categories of either ``cat`` or ``dog`` (or ``neither``). We can use ``one-hot encoding`` to
convert these categories to a vector of 1's and 0's:

.. code-block::

    dog := [1 0]
    cat := [0 1]
    neither := [0 0]


+--------------+------------+-------------+
| input        | target     | prediction  |
+==============+============+=============+
| image of dog | [1 0]      | [1 0]       |
+--------------+------------+-------------+
| image of cat | [0 1]      | [1 0]       |
+--------------+------------+-------------+
| image of rat | [0 0]      | [0 0]       |
+--------------+------------+-------------+

You can see that our network as incorrectly classified our second image, the image of the cat, as
a dog. How can we state this mathematically?

The goal of the loss function is to measure how wrong the predictions are. A very simple loss
function for this example might be simply summing the vector differences:


.. admonition:: Loss Function Example: Mean-Squared Error
    :class: example

    Let :math:`x^{(i)}` be the :math:`i^{th}` input image.

    Let :math:`\vec{y}^{(i)}` be the :math:`i^{th}` K-dimensional target vector.

    Let :math:`\vec{p}^{(i)}` be the :math:`i^{th}` K-dimensional predicted vector.

    Then we can define a loss function :math:`L` as follows:

    .. math::

        L = \frac{1}{N}\sum_{i=1}^{N}(|\vec{y}^{(i)} - \vec{p}^{(i)}|)^2

This example loss function is known as the mean-squared error function. There are many different
types of loss functions that you can choose for your neural network. Below, we will go over
some of the loss functions used in ``ivadomed``.


Dice Loss
---------

The Dice Loss is based off of the Sørensen–Dice coefficient. It is often used in 2D and 3D
image segmentation, and was introduced in 2016 in the paper:
`V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation <https://arxiv.org/pdf/1606.04797.pdf>`_

.. admonition:: Dice Loss
    :class: example

    Let :math:`p_i \in [0, 1]` be the predicted value for voxel :math:`i`.

    Let :math:`g_i \in [0, 1]` be the target (ground truth) value for voxel :math:`i`.

    Then the dice score for one sample is:

    .. math::

        D = \frac{2 \sum_{i=1}^{N} p_i g_i}{\sum_{i=1}^{N} p_i^2 + \sum_{i=1}^{N} g_i^2}

    where :math:`i \in \mathbb{Z}^+` ranges over the N voxels.


Binary Cross Entropy
--------------------

Binary Cross Entropy is often used for classification problems involving a binary category. For
example, identifying whether or not your the image of your face matches the stored image in the
Face ID software on your iPhone.

.. admonition:: Binary Cross Entropy
    :class: example

    Let :math:`p_i \in [0, 1]` be the predicted value for sample :math:`i`.

    Let :math:`y_i \in [0, 1]` be the target (ground truth) value for sample :math:`i`.

    Then the binary cross entropy loss is:

    .. math::

        L = \frac{-1}{N} \sum_{i=1}^{N} y_i \ln(p_i) + (1 - y_i)\ln(1 - p_i)

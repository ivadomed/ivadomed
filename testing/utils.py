#!/usr/bin/env python
# -*- coding: utf-8

import matplotlib.pyplot as plt
import matplotlib


def plot_transformed_sample(before, after, list_title=[]):
    """Utils tool to plot sample before and after transform, for debugging.

    Args:
        before: sample before transform.
        after: sample after transform.
    """
    if len(list_title) == 0:
        list_title = ['Sample before transform', 'Sample after transform']

    matplotlib.use('TkAgg')
    plt.interactive(False)
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(before, interpolation='nearest', aspect='auto')
    plt.title(list_title[0])

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(after, interpolation='nearest', aspect='auto')
    plt.title(list_title[1])

    plt.show()

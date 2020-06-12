#!/usr/bin/env python
# -*- coding: utf-8

import matplotlib
import matplotlib.pyplot as plt


def plot_transformed_sample(before, after, list_title=[], fname_out="", cmap="jet"):
    """Utils tool to plot sample before and after transform, for debugging.

    Args:
        before np.array: sample before transform.
        after np.array: sample after transform.
        list_title list of strings: sub titles of before and after, resp.
        fname_out string: output filename where the plot is saved if provided.
        cmap string: Matplotlib colour map.
    """
    if len(list_title) == 0:
        list_title = ['Sample before transform', 'Sample after transform']

    matplotlib.use('TkAgg')
    plt.interactive(False)
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(before, interpolation='nearest', cmap=cmap)
    plt.title(list_title[0], fontsize=20)

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(after, interpolation='nearest', cmap=cmap)
    plt.title(list_title[1], fontsize=20)

    if fname_out:
        plt.savefig(fname_out)
    else:
        plt.show()

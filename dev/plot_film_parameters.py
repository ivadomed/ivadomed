import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_histogram(data, layer_no, fname_out):
    bins = np.linspace(-6, 6, 100)

    fig = plt.figure(figsize=(12, 12))
    plt.title(f'Layer {layer_no}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.hist(data.ravel(), bins)
    fig.savefig(fname_out)


def visualize_pca(data, layer_no, num_batch, fname_out):
    pca_df = pd.DataFrame()

    pca = PCA(n_components=3)

    for i in range(num_batch):
        pca_result = pca.fit_transform(data[i])
        pca_df2 = pd.DataFrame()
        pca_df2['pca1'] = pca_result[:, 0]
        pca_df2['pca2'] = pca_result[:, 1]
        pca_df2['pca3'] = pca_result[:, 2]
        pca_df2['batch'] = 'batch ' + str(i + 1)
        pca_df = pd.concat([pca_df, pca_df2])

    # Create the graph and save it
    fig = plt.figure(figsize=(16,10))
    plt.title(f"PCA for gammas: Layer {layer_no}")
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="batch",
        palette=sns.color_palette("hls", num_batch),
        data=pca_df,
        legend="full",
        alpha=1)
    fig.savefig(fname_out)


def visualize_tsne(data, num_batch, fname_out):
    tsne_df = pd.DataFrame()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

    for i in range(num_batch):
        tsne_results = tsne.fit_transform(data[i])
        tsne_df2 = pd.DataFrame()

        tsne_df2['tsne-2d-one'] = tsne_results[:, 0]
        tsne_df2['tsne-2d-two'] = tsne_results[:, 1]
        tsne_df2['batch'] = 'batch ' + str(i + 1)

        tsne_df = pd.concat([tsne_df, tsne_df2])

    print('t-SNE done!')

    # Visualize
    fig = plt.figure()
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="batch",
        palette=sns.color_palette("hls", num_batch),
        data=tsne_df,
        legend="full",
        alpha=1
    )
    plt.xlim(-260, 500)
    plt.ylim(-500, 500)
    fig.savefig(fname_out)

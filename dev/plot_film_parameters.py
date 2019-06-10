import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

def visualize_pca(film_parameters, layer_no, batch_no, outdir_name):
    pca_df = pd.DataFramee()

    pca = PCA(n_components=3)

    for i in range(batch_no):
        pca_result = pca.fit_transform(film_parameters[i])
        pca_df2 = pd.DataFrame()
        pca_df2['pca1'] = pca_result[:, 0]
        pca_df2['pca2'] = pca_result[:, 1]
        pca_df2['pca3'] = pca_result[:, 2]
        pca_df2['batch'] = 'batch ' + str(i + 1)
        pca_df = pd.concat([pca_df, pca_df2])


    # Create the graph and save it
    fig = plt.figure()
    plt.title(f"PCA for gammas: {layer_no} layer")
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="batch",
        palette=sns.color_palette("hls", num_batch),
        data=pca_df,
        legend="full",
        alpha=1)
    fig.savefig(outdir_name)

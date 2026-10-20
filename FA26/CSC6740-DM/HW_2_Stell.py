import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import umap
import time


# Question 3

file_path = Path(__file__).resolve().parent
Q3_dataset_path = file_path / "penguins.csv"
Q3_data = pd.read_csv(Q3_dataset_path)
Q3_output_path = file_path / "HW_2_output"
Q3_output_path.mkdir(parents=True, exist_ok=True)


def Q3_main():
    # 3.a

    # 3.a.i
    print("There are {} rows in the dataset.".format(Q3_data.shape[0]))
    # 3.a.ii
    print("Per column, these are the counts of missing value: \n{}".format(Q3_data.isnull().sum()))
    # 3.a.iii
    print("The number of penguins per species is: \n{}".format(Q3_data['species'].value_counts()))

    # 3.b

    # apply the per-species mean of each column to fill the missing values for numerical columns while categorical columns will be filled with the mode of the column
    for column in Q3_data.columns:
        if Q3_data[column].dtype == 'float64' or Q3_data[column].dtype == 'int64':
            Q3_data[column] = Q3_data[column].fillna(Q3_data.groupby('species')[column].transform('mean'))
        else:
            Q3_data[column] = Q3_data[column].fillna(Q3_data[column].mode()[0])


    # 3.c

    # written answer (approach, and what is lost or risked by (i) dropping rows,
    # (ii) imputing with the global mean, (iii) imputing with the per-species mean)


    # 3.d

    # only the four numeric features go into the PCA
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    Q3_features = Q3_data[features]

    print("The variance of each feature before standardizing is: \n{}".format(Q3_features.var()))

    # to compute the PCA, we need to standardize.

    Q3_data_standardized = (Q3_features - Q3_features.mean()) / Q3_features.std()

    print("The variance of each feature after standardizing is: \n{}".format(Q3_data_standardized.var()))


    # 3.e

    pca = PCA()
    pca.fit(Q3_data_standardized)

    def plot_scree(pca, title = 'Scree Plot', filename = 'Q3_scree_plot.png'):

        components = range(1, len(pca.explained_variance_ratio_) + 1)
        plt.bar(components, pca.explained_variance_ratio_, label='Individual')
        plt.plot(components, pca.explained_variance_ratio_.cumsum(), marker='o', color='red', label='Cumulative')
        plt.axhline(0.9, color='gray', linestyle='--', label='90%')
        plt.title(title)
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.xticks(components)
        plt.legend()
        plt.savefig(Q3_output_path / filename, dpi=150)
        plt.close()

    # 3.e.i
    plot_scree(pca)

    print("The explained variance ratio per component is: \n{}".format(pca.explained_variance_ratio_))
    print("The cumulative explained variance ratio is: \n{}".format(pca.explained_variance_ratio_.cumsum()))

    # 3.e.ii

    # correlation matrix before PCA (standardized features) and after PCA (PC scores)
    Q3_pc_scores = pd.DataFrame(pca.transform(Q3_data_standardized), columns=['PC1', 'PC2', 'PC3', 'PC4'])
    feature_corr = Q3_data_standardized.corr()
    pc_corr = Q3_pc_scores.corr()

    print("The correlation matrix of the standardized features is: \n{}".format(feature_corr.round(3)))
    print("The correlation matrix of the PC scores is: \n{}".format(pc_corr.round(3)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, corr, title in zip(axes, [feature_corr, pc_corr], ['Standardized Features', 'PC Scores']):
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, "{:.3f}".format(abs(corr.iloc[i, j]) if abs(corr.iloc[i, j]) < 1e-3 else corr.iloc[i, j]), ha='center', va='center')
        ax.set_title('Correlation: {}'.format(title))
    fig.colorbar(im, ax=axes)
    fig.savefig(Q3_output_path / 'Q3_correlation_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3.e.iii

    # written answer (off-diagonal PC correlations vs. feature correlations, and the most correlated feature pairs)


    # 3.f

    # number of components needed to explain 90% of the variance
    print("Standardized: {} components are needed to explain 90% of the variance.".format((pca.explained_variance_ratio_.cumsum() < 0.9).sum() + 1))

    pca_unstandardized = PCA()
    pca_unstandardized.fit(Q3_features)

    print("Unstandardized cumulative explained variance ratio is: \n{}".format(pca_unstandardized.explained_variance_ratio_.cumsum()))
    print("Unstandardized: {} components are needed to explain 90% of the variance.".format((pca_unstandardized.explained_variance_ratio_.cumsum() < 0.9).sum() + 1))
    print("Unstandardized PC1 loadings are: \n{}".format(pd.Series(pca_unstandardized.components_[0], index=features)))

    plot_scree(pca_unstandardized, title='Scree Plot (unstandardized)', filename='Q3_scree_plot_unstandardized.png')


    # 3.g

    loadings = pd.DataFrame(pca.components_.T, index=features, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    print("The PCA loadings are: \n{}".format(loadings.round(3)))

    # scatter plot of every pair of PCs, colored by species
    for i in range(4):
        for j in range(i + 1, 4):
            for species in Q3_data['species'].unique():
                mask = Q3_data['species'] == species
                plt.scatter(Q3_pc_scores.iloc[:, i][mask], Q3_pc_scores.iloc[:, j][mask], label=species, alpha=0.7)
            plt.title('PCA Projection (PC{} vs PC{})'.format(i + 1, j + 1))
            plt.xlabel('PC{} ({:.1%} of variance)'.format(i + 1, pca.explained_variance_ratio_[i]))
            plt.ylabel('PC{} ({:.1%} of variance)'.format(j + 1, pca.explained_variance_ratio_[j]))
            plt.legend()
            plt.savefig(Q3_output_path / 'Q3_pca_projection_PC{}_PC{}.png'.format(i + 1, j + 1), dpi=150)
            plt.close()

# Question 4

digits = load_digits()
Q4_X, Q4_y = digits.data, digits.target # X: (1797, 64), y: (1797,) digit labels 0-9
Q4_output_path = file_path / "HW_2_output"


def Q4_main():
    # 4.a

    print("The shape of the data X is {} and the shape of the labels y is {}.".format(Q4_X.shape, Q4_y.shape))
    print("There are {} samples, each with {} features (an 8x8 image flattened to {} pixels).".format(Q4_X.shape[0], Q4_X.shape[1], 8 * 8))
    print("The pixel values range from {:g} to {:g}.".format(Q4_X.min(), Q4_X.max()))
    print("There are {} classes: {}".format(len(np.unique(Q4_y)), np.unique(Q4_y)))
    print("The number of samples per class is: \n{}".format(pd.Series(Q4_y).value_counts().sort_index()))

    # reshape the first 10 samples back to 8x8 and show them as images
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for ax, image, label in zip(axes.ravel(), Q4_X[:10], Q4_y[:10]):
        ax.imshow(image.reshape(8, 8), cmap='gray_r')
        ax.set_title('Label: {}'.format(label))
        ax.axis('off')
    fig.suptitle('Sample Digits (8x8)')
    fig.savefig(Q4_output_path / 'Q4_sample_digits.png', dpi=150)
    plt.close(fig)


    # 4.b

    pca = PCA(n_components=50)
    Q4_pc_scores = pca.fit_transform(Q4_X)

    components = range(1, len(pca.explained_variance_ratio_) + 1)
    plt.figure(figsize=(10, 5))
    plt.bar(components, pca.explained_variance_ratio_, label='Individual')
    plt.plot(components, pca.explained_variance_ratio_.cumsum(), marker='o', markersize=3, color='red', label='Cumulative')
    plt.axhline(0.9, color='gray', linestyle='--', label='90%')
    plt.title('Scree Plot: Digits (50 components)')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.savefig(Q4_output_path / 'Q4_scree_plot.png', dpi=150)
    plt.close()

    print("The explained variance ratio of the first 10 components is: \n{}".format(pca.explained_variance_ratio_[:10].round(4)))
    print("The cumulative explained variance ratio at 2, 5, 10, 20, 40, 50 components is: \n{}".format(
        pca.explained_variance_ratio_.cumsum()[[1, 4, 9, 19, 39, 49]].round(4)))
    print("{} components are needed to explain 90% of the variance.".format((pca.explained_variance_ratio_.cumsum() < 0.9).sum() + 1))


    # 4.c

    plt.figure(figsize=(8, 6.5))
    scatter = plt.scatter(Q4_pc_scores[:, 0], Q4_pc_scores[:, 1], c=Q4_y, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Digit')
    plt.title('PCA Projection of Digits (PC1 vs PC2)')
    plt.xlabel('PC1 ({:.1%} of variance)'.format(pca.explained_variance_ratio_[0]))
    plt.ylabel('PC2 ({:.1%} of variance)'.format(pca.explained_variance_ratio_[1]))
    plt.savefig(Q4_output_path / 'Q4_pca_projection_PC1_PC2.png', dpi=150)
    plt.close()

    print("The first 2 components explain {:.2%} of the variance, compared to {:.2%} for all 50.".format(
        pca.explained_variance_ratio_[:2].sum(), pca.explained_variance_ratio_.sum()))


    # 4.d

    # pick 5 random images (seeded so the same ones are picked each run)
    rng = np.random.default_rng(6740)
    sample_index = rng.choice(len(Q4_X), size=5, replace=False)
    k_values = [5, 20, 40]

    fig, axes = plt.subplots(5, len(k_values) + 1, figsize=(8, 10))
    for col, k in enumerate([None] + k_values):
        if k is None:
            images = Q4_X[sample_index]
            title = 'Original'
        else:
            pca_k = PCA(n_components=k)
            pca_k.fit(Q4_X)
            images = pca_k.inverse_transform(pca_k.transform(Q4_X[sample_index]))
            title = 'k = {}'.format(k)
            mse = ((images - Q4_X[sample_index]) ** 2).mean()
            print("k = {}: explained variance {:.1%}, reconstruction MSE on the 5 samples = {:.3f}".format(
                k, pca_k.explained_variance_ratio_.sum(), mse))
        for row in range(5):
            ax = axes[row, col]
            ax.imshow(images[row].reshape(8, 8), cmap='gray_r', vmin=0, vmax=16)
            ax.axis('off')
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.text(-1.5, 3.5, 'Label: {}'.format(Q4_y[sample_index[row]]), ha='right', va='center')
    fig.suptitle('PCA Reconstructions from the top k components')
    fig.savefig(Q4_output_path / 'Q4_reconstructions.png', dpi=150)
    plt.close(fig)

# Question 5

# PCA-reduced (50-dim) digits, shared by Question 5 and Question 6
Q5_X_pca = PCA(n_components=50).fit_transform(Q4_X)
Q5_output_path = file_path / "HW_2_output"


def plot_embedding(ax, embedding, title):
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=Q4_y, cmap='tab10', s=4, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter


def Q5_main():
    # 5.a

    # t-SNE with perplexity 5 on the raw 64-dim data and on the PCA-reduced 50-dim data
    start = time.perf_counter()
    tsne_raw = TSNE(n_components=2, perplexity=5, random_state=0).fit_transform(Q4_X)
    time_raw = time.perf_counter() - start

    start = time.perf_counter()
    tsne_pca = TSNE(n_components=2, perplexity=5, random_state=0).fit_transform(Q5_X_pca)
    time_pca = time.perf_counter() - start

    print("t-SNE runtime on raw data (64-dim): {:.2f} s".format(time_raw))
    print("t-SNE runtime on PCA-reduced data (50-dim): {:.2f} s".format(time_pca))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_embedding(axes[0], tsne_raw, 'Raw data (64-dim), {:.2f} s'.format(time_raw))
    scatter = plot_embedding(axes[1], tsne_pca, 'PCA-reduced (50-dim), {:.2f} s'.format(time_pca))
    fig.colorbar(scatter, ax=axes, ticks=range(10), label='Digit')
    fig.suptitle('t-SNE, perplexity = 5')
    fig.savefig(Q5_output_path / 'Q5_tsne_raw_vs_pca.png', dpi=150)
    plt.close(fig)


    # 5.b

    perplexities = [5, 30, 50, 100]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, perplexity in zip(axes.ravel(), perplexities):
        embedding = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(Q5_X_pca)
        scatter = plot_embedding(ax, embedding, 'Perplexity = {}'.format(perplexity))
    fig.colorbar(scatter, ax=axes, ticks=range(10), label='Digit')
    fig.suptitle('t-SNE perplexity sweep (PCA-reduced data)')
    fig.savefig(Q5_output_path / 'Q5_tsne_perplexity_sweep.png', dpi=150)
    plt.close(fig)


    # 5.c

    # same perplexity, two different random seeds
    # init='random' so the seed sets the starting layout (the default init='pca' is the same for every seed)
    seeds = [137, 29]
    embeddings = [TSNE(n_components=2, perplexity=30, init='random', random_state=seed).fit_transform(Q5_X_pca) for seed in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, seed, embedding in zip(axes, seeds, embeddings):
        scatter = plot_embedding(ax, embedding, 'Random seed = {}'.format(seed))
    fig.colorbar(scatter, ax=axes, ticks=range(10), label='Digit')
    fig.suptitle('t-SNE, perplexity = 30, random init, two random seeds')
    fig.savefig(Q5_output_path / 'Q5_tsne_seeds.png', dpi=150)
    plt.close(fig)

    # compare the distances between digit cluster centers in the two embeddings
    upper = np.triu_indices(10, k=1)
    center_distances = []
    for embedding in embeddings:
        centers = np.array([embedding[Q4_y == digit].mean(axis=0) for digit in range(10)])
        center_distances.append(np.linalg.norm(centers[:, None] - centers[None, :], axis=2)[upper])
    print("Correlation of the 45 cluster-center distances between seed {} and seed {}: {:.3f}".format(seeds[0], seeds[1],
        np.corrcoef(center_distances[0], center_distances[1])[0, 1]))
    for seed, distances in zip(seeds, center_distances):
        closest = distances.argmin()
        print("Seed {}: closest digit clusters are {} and {}".format(seed, upper[0][closest], upper[1][closest]))


# Question 6

Q6_output_path = file_path / "HW_2_output"


def Q6_main():
    # UMAP on the same PCA-reduced data, over a grid of n_neighbors and min_dist
    n_neighbors_values = [5, 15, 50, 200]
    min_dist_values = [0.0, 0.1, 0.5, 0.9]

    fig, axes = plt.subplots(len(n_neighbors_values), len(min_dist_values), figsize=(16, 16))
    for row, n_neighbors in enumerate(n_neighbors_values):
        for col, min_dist in enumerate(min_dist_values):
            embedding = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                                  random_state=0).fit_transform(Q5_X_pca)
            # label the columns by min_dist and the rows by n_neighbors so the titles don't overlap
            scatter = plot_embedding(axes[row, col], embedding, 'min_dist = {}'.format(min_dist) if row == 0 else '')
            if col == 0:
                axes[row, col].set_ylabel('n_neighbors = {}'.format(n_neighbors), fontsize=12)
    fig.colorbar(scatter, ax=axes, ticks=range(10), label='Digit', shrink=0.5)
    fig.suptitle('UMAP grid (PCA-reduced data): rows = n_neighbors, columns = min_dist')
    fig.savefig(Q6_output_path / 'Q6_umap_grid.png', dpi=150)
    plt.close(fig)

# run

if __name__ == "__main__":
    Q3_main()
    Q4_main()
    Q5_main()
    Q6_main()

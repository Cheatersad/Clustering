import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering, AffinityPropagation, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import estimate_bandwidth

def assign_cluster_labels(data, labels):
    cluster_stats = data.groupby(labels).agg(['mean', 'std'])
    cluster_criticality = {}

    for cluster in cluster_stats.index.unique():
        cluster_data = cluster_stats.loc[cluster]
        criticality_score = (
            cluster_data['subscriber_count']['mean'] +
            cluster_data['N2_s1mme_total']['mean'] +
            cluster_data['N12_s6ad_Total']['mean'] +
            cluster_data['N11_S11_Total']['mean'] +
            cluster_data['NT_AMF_PCMD']['mean'] +
            cluster_data['Total_PCMD']['mean']
        )
        cluster_criticality[cluster] = criticality_score

    sorted_clusters = sorted(cluster_criticality, key=cluster_criticality.get, reverse=True)
    criticality_labels = ['Blocker', 'Critical', 'Major', 'Trivial']
    cluster_label_mapping = dict(zip(sorted_clusters, criticality_labels))

    return labels.map(cluster_label_mapping)

def main():
    # Load data
    data = pd.read_csv('output_testing_new.csv')

    # Select specific columns
    selected_columns = [0, 1, 2, 3, 4, 13, 16, 29, 19, 34, 25, 37, 40, 41, 42, 43]
    selected_data = data.iloc[:, selected_columns].copy()

    # Create new columns using .loc[]
    selected_data.loc[:, 'N2_s1mme_total'] = selected_data.iloc[:, 6] + selected_data.iloc[:, 7]
    selected_data.loc[:, 'N12_s6ad_Total'] = selected_data.iloc[:, 8] + selected_data.iloc[:, 9]
    selected_data.loc[:, 'N11_S11_Total'] = selected_data.iloc[:, 10] + selected_data.iloc[:, 11]
    selected_data.loc[:, 'Total_PCMD'] = selected_data.iloc[:, 13] + selected_data.iloc[:, 14] + selected_data.iloc[:, 15]

    # Select specific columns for clustering
    selected_columns = ['subscriber_count', 'N2_s1mme_total', 'N12_s6ad_Total', 'N11_S11_Total', 'NT_AMF_PCMD', 'Total_PCMD']
    clustering_data = selected_data[selected_columns]

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Dimensionality reduction with PCA
    pca = PCA(n_components=0.95)
    pca_result = pca.fit_transform(scaled_data)

    # Dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_components = tsne.fit_transform(pca_result)
    tsne_df = pd.DataFrame(data=tsne_components, columns=['TSNE1', 'TSNE2'])

    bandwidth=estimate_bandwidth(tsne_df, quantile=0.5, n_samples=5)
    print(bandwidth)
    
    # Clustering algorithms
    clustering_algorithms = [
        KMeans(n_clusters=4, random_state=0),
        AgglomerativeClustering(n_clusters=4),
        DBSCAN(eps=0.5, min_samples=5),
        MeanShift(bandwidth=bandwidth, bin_seeding=True),
        SpectralClustering(n_clusters=4, random_state=0),
        AffinityPropagation(),
        OPTICS(min_samples=5),
        Birch(n_clusters=4),
        GaussianMixture(n_components=4, random_state=0)
    ]

    # Perform clustering and plot results
    plt.figure(figsize=(20, 15))

    # Create a DataFrame to store the cluster labels for each algorithm
    cluster_labels_df = pd.DataFrame()

    for i, algorithm in enumerate(clustering_algorithms):
        # Perform clustering
        if hasattr(algorithm, 'fit_predict'):
            labels = algorithm.fit_predict(tsne_df[['TSNE1', 'TSNE2']])
        else:
            labels = algorithm.fit(tsne_df[['TSNE1', 'TSNE2']]).predict(tsne_df[['TSNE1', 'TSNE2']])
        
        # Assign cluster labels based on criticality
        labels = assign_cluster_labels(clustering_data, pd.Series(labels))
        
        # Add the cluster labels to the DataFrame
        cluster_labels_df[algorithm.__class__.__name__] = labels
        
        # Handle NaN values in labels
        if labels.isnull().any():
            print(f"Warning: {algorithm.__class__.__name__} produced NaN labels. Skipping silhouette score calculation.")
            silhouette_avg = np.nan
        else:
            # Evaluate clustering using silhouette score
            silhouette_avg = silhouette_score(tsne_df[['TSNE1', 'TSNE2']], labels)
        
        # Plot the clusters
        plt.subplot(3, 3, i + 1)
        sns.scatterplot(x='TSNE1', y='TSNE2', hue=labels, palette='viridis', s=100, alpha=0.8, data=tsne_df)
        plt.title(f'{algorithm.__class__.__name__} (Silhouette: {silhouette_avg:.2f})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Cluster Criticality', loc='upper right')

    plt.tight_layout()
    plt.show()

    # Combine the first 5 columns with the clustered data and cluster labels
    output_data = pd.concat([selected_data.iloc[:, :5], clustering_data, tsne_df, cluster_labels_df], axis=1)

    # Save the clustered data
    output_data.to_csv('clustered_output_all_algorithms_improved.csv', index=False)
    print("Output saved to 'clustered_output_all_algorithms_improved.csv'")

if __name__ == '__main__':
    main()

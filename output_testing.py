import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

def assign_cluster_labels(data, labels):
    cluster_stats = data.groupby(labels).agg(['mean', 'std'])
    cluster_criticality = {}

    for cluster in cluster_stats.index.unique():
        cluster_data = cluster_stats.loc[cluster]
        criticality_score = (
            cluster_data['subscriber_count']['mean'] +
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
    selected_columns = [0, 1, 2, 3, 4, 13, 40, 41, 42, 43]
    selected_data = data.iloc[:, selected_columns].copy()

    selected_data.loc[:, 'Total_PCMD'] = selected_data.iloc[:, 6] + selected_data.iloc[:, 7] + selected_data.iloc[:, 8]

    # Select specific columns for clustering
    selected_columns = ['subscriber_count', 'Total_PCMD']
    clustering_data = selected_data[selected_columns]

    # Handle outliers using RobustScaler
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_components = tsne.fit_transform(scaled_data)
    tsne_df = pd.DataFrame(data=tsne_components, columns=['TSNE1', 'TSNE2'])

    # Spectral embedding with increased n_neighbors
    spectral_embedding = SpectralEmbedding(n_components=2, n_neighbors=10, random_state=0)
    spectral_components = spectral_embedding.fit_transform(scaled_data)
    spectral_df = pd.DataFrame(data=spectral_components, columns=['Spectral1', 'Spectral2'])

    # Clustering algorithms with different hyperparameters
    clustering_algorithms = [
        KMeans(n_clusters=4, random_state=0),
        AgglomerativeClustering(n_clusters=4),
        SpectralClustering(n_clusters=4, affinity='nearest_neighbors', n_neighbors=10, random_state=0),
        GaussianMixture(n_components=4, random_state=0)
    ]

    # Perform clustering and plot results
    plt.figure(figsize=(20, 15))

    # Create a DataFrame to store the cluster labels for each algorithm
    cluster_labels_df = pd.DataFrame()

    for i, algorithm in enumerate(clustering_algorithms):
        # Perform clustering
        if hasattr(algorithm, 'fit_predict'):
            labels = algorithm.fit_predict(spectral_df)
        else:
            labels = algorithm.fit(spectral_df).predict(spectral_df)
        
        # Assign cluster labels based on criticality
        labels = assign_cluster_labels(clustering_data, pd.Series(labels))
        
        # Add the cluster labels to the DataFrame
        cluster_labels_df[algorithm.__class__.__name__] = labels
        
        # Handle NaN values in labels
        if labels.isnull().any():
            print(f"Warning: {algorithm.__class__.__name__} produced NaN labels. Skipping silhouette score calculation.")
            silhouette_avg = np.nan
        else:
            # Check the number of unique labels
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                print(f"Warning: {algorithm.__class__.__name__} produced only one cluster label. Skipping silhouette score calculation.")
                silhouette_avg = np.nan
            else:
                # Evaluate clustering using silhouette score
                silhouette_avg = silhouette_score(spectral_df, labels)
        
        # Plot the clusters
        plt.subplot(2, 2, i + 1)
        sns.scatterplot(x='Spectral1', y='Spectral2', hue=labels, palette='viridis', s=100, alpha=0.8, data=spectral_df)
        plt.title(f'{algorithm.__class__.__name__} (Silhouette: {silhouette_avg:.2f})')
        plt.xlabel('Spectral Component 1')
        plt.ylabel('Spectral Component 2')
        plt.legend(title='Cluster Criticality', loc='upper right')

    plt.tight_layout()
    plt.show()

    # Combine the first 5 columns with the clustered data and cluster labels
    output_data = pd.concat([selected_data.iloc[:, :5], clustering_data, spectral_df, cluster_labels_df], axis=1)

    # Save the clustered data
    output_data.to_csv('clustered_output_all_algorithms_improved.csv', index=False)
    print("Output saved to 'clustered_output_all_algorithms_improved.csv'")

if __name__ == '__main__':
    main()

#!/usr/bin/env python
# coding: utf-8

"""
This script performs clustering analysis on telecommunication network data.
It reads data from a CSV file, preprocesses the data, performs dimensionality
reduction using t-SNE, applies various clustering algorithms, and visualizes
the clustering results.

Author: Your Name
Date: Current Date
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture

def preprocess_data(input_file):
    """
    Preprocess the data by reading the CSV file, creating new columns,
    and selecting the required columns.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        pandas.DataFrame: Preprocessed data.
    """
    df = pd.read_csv(input_file)
    df['N2_s1mme_total'] = df['N2_total'] + df['s1mme_total']
    df['N11_s11_total'] = df['N11_total'] + df['s11_total']
    df['N12_s6ad_total'] = df['N12_total'] + df['s6ad_total']
    df['PCMD_total'] = df['NT_AMF_PCMD'] + df['NT_SGSN_PCMD'] + df['NT_MME_PCMD']

    need_columns = ['Node Name', 'subscriber_count', 'N2_s1mme_total', 'PCMD_total']
    preprocessed_data = df[need_columns]

    return preprocessed_data

def perform_clustering(data, num_clusters=6):
    """
    Perform dimensionality reduction and clustering on the input data.

    Args:
        data (pandas.DataFrame): Input data for clustering.
        num_clusters (int, optional): Number of clusters. Default is 6.

    Returns:
        tuple: (pandas.DataFrame, pandas.DataFrame)
            Tuple containing the input data with clustering labels added,
            and the t-SNE components.
    """
    # Drop 'Node Name' column for clustering
    cluster_data = data.drop(['Node Name'], axis=1)

    # Handle outliers using RobustScaler
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    # Dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_components = tsne.fit_transform(scaled_data)
    tsne_df = pd.DataFrame(data=tsne_components, columns=['TSNE1', 'TSNE2'])

    # Initialize clustering algorithms
    clustering_algorithms = [
        ("KMeans", KMeans(n_clusters=num_clusters)),
        ("Agglomerative", AgglomerativeClustering(n_clusters=num_clusters)),
        ("Spectral", SpectralClustering(n_clusters=num_clusters)),
        ("Birch", Birch(n_clusters=num_clusters)),
        ("GMM", GaussianMixture(n_components=num_clusters))
    ]

    # Perform clustering and append results to DataFrame
    for name, algorithm in clustering_algorithms:
        algorithm.fit(tsne_df)
        if hasattr(algorithm, 'labels_'):
            data[name] = algorithm.labels_
        else:
            # For GMM, assign labels based on highest probability
            probabilities = algorithm.predict_proba(tsne_df)
            labels = np.argmax(probabilities, axis=1)
            data[name] = labels

    return data, tsne_df

def map_cluster_labels(data, cluster_col, label_names):
    """
    Map cluster labels to meaningful names based on the mean PCMD_total value.

    Args:
        data (pandas.DataFrame): Input data with clustering labels.
        cluster_col (str): Name of the column containing cluster labels.
        label_names (list): List of label names to assign.

    Returns:
        pandas.DataFrame: Input data with cluster labels mapped to meaningful names.
    """
    cluster_means = data.groupby(cluster_col)['PCMD_total'].mean()
    cluster_means_int = cluster_means.astype(int)
    cluster_means_int_sorted = cluster_means_int.sort_values(ascending=False)

    cluster_df = pd.DataFrame({cluster_col: cluster_means_int_sorted.index})
    cluster_df['Label'] = label_names

    cluster_label_map = dict(zip(cluster_df[cluster_col], cluster_df['Label']))
    data[f'{cluster_col}_mapped'] = data[cluster_col].map(cluster_label_map)

    return data

def visualize_clusters(data, label_col):
    """
    Visualize the clustering results using a scatter plot.

    Args:
        data (pandas.DataFrame): Input data with clustering labels and t-SNE components.
        label_col (str): Name of the column containing the cluster labels to visualize.
    """
    plot_df = data[['TSNE1', 'TSNE2', label_col]]

    # Separate data points based on label values
    label_groups = plot_df.groupby(label_col)

    # Plotting
    plt.figure(figsize=(8, 6))
    for label, group in label_groups:
        plt.scatter(group['TSNE1'], group['TSNE2'], label=label)

    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.title('TSNE1 vs TSNE2')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to run the clustering analysis pipeline.
    """
    input_file = 'output_testing_new.csv'
    preprocessed_data = preprocess_data(input_file)
    clustered_data, tsne_df = perform_clustering(preprocessed_data)

    label_names = ["Blocker", "Critical", "Major", "Major", "Major", "Trivial"]

    for cluster_col in ["KMeans", "Agglomerative", "Spectral", "Birch", "GMM"]:
        clustered_data = map_cluster_labels(clustered_data, cluster_col, label_names)

    # Merge t-SNE components back into the clustered data
    clustered_data = pd.concat([clustered_data, tsne_df], axis=1)

    # Save the clustered data to a CSV file
    clustered_data.to_csv('clustering_results.csv', index=True)
    print("Clustering results saved to clustering_results.csv")

    # Visualize the KMeans clustering results
    visualize_clusters(clustered_data, 'KMeans_mapped')

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def preprocess_data(input_file):
    df = pd.read_csv(input_file)
    df['N2_s1mme_total'] = df['N2_total'] + df['s1mme_total']
    df['N11_s11_total'] = df['N11_total'] + df['s11_total']
    df['N12_s6ad_total'] = df['N12_total'] + df['s6ad_total']
    df['PCMD_total'] = df['NT_AMF_PCMD'] + df['NT_SGSN_PCMD'] + df['NT_MME_PCMD']

    need_columns = ['Node Name', 'subscriber_count', 'N2_s1mme_total', 'N11_s11_total', 'N12_s6ad_total', 'PCMD_total']
    return df[need_columns]

def perform_clustering(data, num_clusters=6):
    cluster_data = data.drop(['Node Name'], axis=1)
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_components = tsne.fit_transform(scaled_data)
    tsne_df = pd.DataFrame(data=tsne_components, columns=['tsne_1', 'tsne_2'])

    clustering_algorithms = [
        ("KMeans", KMeans(n_clusters=num_clusters)),
        ("Agglomerative", AgglomerativeClustering(n_clusters=num_clusters)),
        ("GMM", GaussianMixture(n_components=num_clusters))
    ]

    for name, algorithm in clustering_algorithms:
        if name == "GMM":
            algorithm.fit(tsne_df)
            labels = algorithm.predict(tsne_df)
        else:
            algorithm.fit(tsne_df)
            labels = algorithm.labels_
        data[name] = labels

    return data, tsne_df

def map_cluster_labels(data, cluster_col):
    labels = ["Blocker", "Critical", "Major", "Minor", "Trivial", "Unknown"]
    cluster_means = data.groupby(cluster_col)['PCMD_total'].mean()
    cluster_means_int = cluster_means.astype(int)
    cluster_means_int_sorted = cluster_means_int.sort_values(ascending=False)

    cluster_df = pd.DataFrame({cluster_col: cluster_means_int_sorted.index})
    cluster_df['Label'] = labels[:len(cluster_df)]

    cluster_label_map = dict(zip(cluster_df[cluster_col], cluster_df['Label']))
    data[f'{cluster_col}_T'] = data[cluster_col].map(cluster_label_map)

    return data

def visualize_clusters(data, label_col):
    plot_df = data[['tsne_1', 'tsne_2', label_col]].copy()
    unique_labels = plot_df[label_col].unique()
    color_map = plt.colormaps['viridis']
    label_colors = {label: color_map(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        group = plot_df[plot_df[label_col] == label]
        ax.scatter(group['tsne_1'], group['tsne_2'], label=label, color=label_colors[label])

    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')
    ax.set_title('Cluster Visualization')
    ax.legend(title='Cluster Labels', loc='upper right')
    ax.grid(True)

    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0, len(unique_labels) - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Cluster Labels')
    cbar.set_ticks(range(len(unique_labels)))
    cbar.set_ticklabels(unique_labels)

    plt.tight_layout()
    plt.show()

def main():
    input_file = 'output_testing_new.csv'
    preprocessed_data = preprocess_data(input_file)
    clustered_data, tsne_df = perform_clustering(preprocessed_data)

    for cluster_col in ["KMeans", "Agglomerative", "GMM"]:
        clustered_data = map_cluster_labels(clustered_data, cluster_col)

    columns_to_consider = ['KMeans_T', 'Agglomerative_T', 'GMM_T']
    resultant_column = clustered_data[columns_to_consider].mode(axis=1)[0]
    clustered_data['resultant_column'] = resultant_column

    clustered_data = pd.concat([clustered_data, tsne_df], axis=1)
    clustered_data.to_csv('clustering_results_improved.csv', index=False)
    print("Clustering results saved to clustering_results_improved.csv")

    visualize_clusters(clustered_data, 'resultant_column')

if __name__ == "__main__":
    main()

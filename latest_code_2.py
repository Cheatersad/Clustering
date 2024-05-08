import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib  # Ensure matplotlib is explicitly imported


def preprocess_data(input_file):
    df = pd.read_csv(input_file)
    df['N2_s1mme_total'] = df['N2_total'] + df['s1mme_total']
    df['N11_s11_total'] = df['N11_total'] + df['s11_total']
    df['N12_s6ad_total'] = df['N12_total'] + df['s6ad_total']
    df['PCMD_total'] = df['NT_AMF_PCMD'] + df['NT_SGSN_PCMD'] + df['NT_MME_PCMD']
    need_columns = ['Node Name', 'subscriber_count', 'N2_s1mme_total', 'PCMD_total']
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

def map_cluster_labels(data, cluster_col, labels):
    cluster_means = data.groupby(cluster_col)[['PCMD_total', 'subscriber_count']].mean()
    cluster_means['combined_mean'] = cluster_means.mean(axis=1)
    cluster_order = cluster_means['combined_mean'].sort_values(ascending=False).index
    label_dict = {}
    for i, cluster in enumerate(cluster_order):
        label_dict[cluster] = labels[i]
    data[f'{cluster_col}_T'] = data[cluster_col].map(label_dict)
    return data

def visualize_clusters(data, tsne_df, label_col):
    plot_df = pd.concat([tsne_df, data[[label_col]]], axis=1)
    unique_labels = plot_df[label_col].unique()
    color_map = matplotlib.colormaps['viridis']
    label_colors = {label: color_map(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in unique_labels:
        group = plot_df[plot_df[label_col] == label]
        ax.scatter(group['tsne_1'], group['tsne_2'], label=label, color=label_colors[label])
    ax.set_xlabel('tsne_1')
    ax.set_ylabel('tsne_2')
    ax.set_title('TSNE 1 vs TSNE 2')
    ax.legend()
    ax.grid(True)
    plt.show()

def main():
    input_file = 'output_testing_new.csv'
    preprocessed_data = preprocess_data(input_file)
    clustered_data, tsne_df = perform_clustering(preprocessed_data)
    labels = ["Blocker", "Critical", "Major", "Major", "Major", "Trivial"]
    for cluster_col in ["KMeans", "Agglomerative", "GMM"]:
        clustered_data = map_cluster_labels(clustered_data, cluster_col, labels)
    columns_to_consider = ['KMeans_T', 'Agglomerative_T', 'GMM_T']
    resultant_column = clustered_data[columns_to_consider].mode(axis=1)[0]
    clustered_data['resultant_column'] = resultant_column
    clustered_data = pd.concat([clustered_data, tsne_df], axis=1)
    clustered_data.to_csv('clustering_results.csv', index=False)
    print("Clustering results saved to clustering_results.csv")
    visualize_clusters(clustered_data, tsne_df, 'resultant_column')

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('output_testing_new.csv')

# Select numeric columns and scale data
numeric_columns = data.select_dtypes(include=['number']).columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numeric_columns])

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_.cumsum()

# Choose number of PCA components that explain at least 80% of the variance
n_components = np.argmax(explained_variance >= 0.80) + 1
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(scaled_data)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

# Determine the optimal number of clusters using Silhouette Scores
silhouettes = []
range_clusters = range(2, 10)
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(principal_df)
    silhouettes.append(silhouette_score(principal_df, labels))

# Choose number of clusters based on highest silhouette score
optimal_clusters = silhouettes.index(max(silhouettes)) + 2  # adjusting index since range starts from 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
principal_df['Cluster'] = kmeans.fit_predict(principal_df)

# Transform centroids back to the original feature space
centroids = scaler.inverse_transform(pca.inverse_transform(kmeans.cluster_centers_))

# Calculate cluster criticality with weighted distances
subscriber_count_index = numeric_columns.get_loc('subscriber_count')
nt_columns_indices = [numeric_columns.get_loc(col) for col in ['NT_AMF_PCMD', 'NT_SGSN_PCMD', 'NT_MME_PCMD']]
subscriber_count_weight = 2
nt_columns_weight = 1.5
other_columns_weight = 1

weighted_distances = []
for i in range(optimal_clusters):
    cluster_indices = principal_df[principal_df['Cluster'] == i].index
    cluster_data = data.loc[cluster_indices, numeric_columns]
    subscriber_count_distance = np.mean(np.abs(cluster_data.iloc[:, subscriber_count_index] - centroids[i, subscriber_count_index])) * subscriber_count_weight
    nt_columns_distance = np.mean(np.linalg.norm(cluster_data.iloc[:, nt_columns_indices] - centroids[i, nt_columns_indices], axis=1)) * nt_columns_weight
    other_columns_distance = np.mean(np.linalg.norm(cluster_data.drop(columns=[numeric_columns[subscriber_count_index]] + [numeric_columns[i] for i in nt_columns_indices]) - np.delete(centroids[i], np.concatenate(([subscriber_count_index], nt_columns_indices))), axis=1)) * other_columns_weight
    weighted_distance = subscriber_count_distance + nt_columns_distance + other_columns_distance
    weighted_distances.append(weighted_distance)

cluster_criticality = np.argsort(weighted_distances)  # Lower distance could be deemed as more critical

# Dynamically generate labels based on criticality
base_labels = ["Low", "Moderate", "High", "Critical"]  # Extend this list as needed
if optimal_clusters > len(base_labels):
    # Extend the labels list if there are more clusters than base labels
    extended_labels = base_labels + [f"Very Critical Level {i + 1}" for i in range(optimal_clusters - len(base_labels))]
else:
    extended_labels = base_labels[:optimal_clusters]

criticality_labels = {cluster_criticality[i]: extended_labels[i] for i in range(optimal_clusters)}

# Update cluster labels
principal_df['Cluster_Label'] = principal_df['Cluster'].map(criticality_labels)

# Merge and save the enhanced dataset
output_data = pd.concat([data, principal_df[['Cluster', 'Cluster_Label']]], axis=1)
output_data.to_csv('clustered_output_optimal.csv', index=False)

# Plotting the final clusters with labels
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Label', style='Cluster_Label', data=principal_df, palette='viridis', s=100, alpha=0.8)
plt.title('Final Clusters Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Annotate each cluster centroid
for i in range(optimal_clusters):
    plt.annotate(criticality_labels[i], (kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
print("Output saved to 'clustered_output_optimal.csv'")

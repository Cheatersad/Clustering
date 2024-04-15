import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = 'output_testing.csv'
data = pd.read_csv(data_path)

# Select numeric columns
numeric_columns = data.select_dtypes(include=['number']).columns

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numeric_columns])

# Identify near-duplicate entries using a statistical threshold
dist_matrix = pairwise_distances(scaled_data)
threshold = 0.05
to_keep = np.array([i for i in range(len(dist_matrix)) if np.all(dist_matrix[i][np.arange(len(dist_matrix)) != i] > threshold)])

# Filter data to remove near-duplicates
filtered_data = data.iloc[to_keep]
filtered_scaled_data = scaled_data[to_keep]

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(filtered_scaled_data)
explained_variance = pca.explained_variance_ratio_.cumsum()

# Determine the number of PCA components to retain based on explained variance
n_components = np.argmax(explained_variance >= 0.80) + 1
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(filtered_scaled_data)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

# Determine the optimal number of clusters using Davies-Bouldin Index, with a threshold to limit clusters
davies_bouldin_scores = []
range_clusters = range(2, 10)
db_threshold = 0.6  # Example threshold for Davies-Bouldin score
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(principal_df)
    db_score = davies_bouldin_score(principal_df, labels)
    davies_bouldin_scores.append(db_score)
    if db_score <= db_threshold:
        break

# Choose the number of clusters with the lowest Davies-Bouldin score below the threshold
optimal_clusters = range_clusters[np.argmin(davies_bouldin_scores)]
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
principal_df['Cluster'] = kmeans.fit_predict(principal_df)

# Calculate cluster criticality
centroids = kmeans.cluster_centers_
distances = [np.mean(np.linalg.norm(principal_df[principal_df['Cluster'] == i][[f'PC{j+1}' for j in range(n_components)]].values - centroids[i], axis=1)) for i in range(optimal_clusters)]
cluster_criticality = np.argsort(distances)

# Dynamically generate labels based on the number of clusters
labels = ["Low", "Moderate", "High", "Critical"]
if optimal_clusters > len(labels):
    additional_labels = [f"Critical Level {i - len(labels) + 1}" for i in range(optimal_clusters - len(labels))]
    extended_labels = labels + additional_labels
else:
    extended_labels = labels[:optimal_clusters]

criticality_labels = {cluster_criticality[i]: extended_labels[i] for i in range(optimal_clusters)}
principal_df['Cluster_Label'] = principal_df['Cluster'].map(criticality_labels)

# Merge and save the enhanced dataset
output_data = pd.concat([filtered_data, principal_df[['Cluster', 'Cluster_Label']]], axis=1)
output_data.to_csv('clustered_output_optimal_test.csv', index=False)

# Plotting the final clusters with labels
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Label', style='Cluster_Label', data=principal_df, palette='viridis', s=100, alpha=0.8)
plt.title('Final Clusters Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.show()
print("Output saved to 'clustered_output_optimal.csv'")

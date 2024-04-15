import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = 'output_testing_new.csv'  # Update the path accordingly
data = pd.read_csv(file_path)

# Select only numeric columns for analysis
numeric_columns = data.select_dtypes(include=['number']).columns
numeric_data = data[numeric_columns]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Apply PCA to reduce dimensionality while retaining 80% of the variance
pca = PCA(n_components=0.80)
principal_components = pca.fit_transform(scaled_data)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# Determine the optimal number of clusters using silhouette scores (example: range 2-10 clusters)
silhouette_scores = []
range_clusters = range(2, 10)
for k in range_clusters:
    clustering = SpectralClustering(n_clusters=k, affinity='rbf', random_state=0, assign_labels='discretize')
    labels = clustering.fit_predict(principal_df)
    silhouette_scores.append(silhouette_score(principal_df, labels))

# Choose the number of clusters with the highest silhouette score
optimal_clusters = range_clusters[silhouette_scores.index(max(silhouette_scores))]
clustering = SpectralClustering(n_clusters=optimal_clusters, affinity='rbf', random_state=0, assign_labels='discretize')
labels = clustering.fit_predict(principal_df)

# Assign clusters back to the principal components DataFrame
principal_df['Cluster'] = labels

# Feature importance for weighting using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(scaled_data, labels)
feature_importances = rf.feature_importances_

# Calculate weighted distances within each cluster
weighted_distances = []
for i in range(optimal_clusters):
    cluster_data = numeric_data[principal_df['Cluster'] == i]
    centroid = cluster_data.mean()
    distance = np.sum((cluster_data - centroid) ** 2 * feature_importances, axis=1).mean()
    weighted_distances.append(distance)

# Determine criticality based on weighted distances
criticality_threshold = np.percentile(weighted_distances, 75)  # top 25% are considered more critical
critical_clusters = [i for i, distance in enumerate(weighted_distances) if distance > criticality_threshold]

# Output the results
print("Optimal number of clusters:", optimal_clusters)
print("Critical clusters based on the top 25% of weighted distances:", critical_clusters)

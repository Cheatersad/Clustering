import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
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

# Prepare for clustering comparisons
range_clusters = range(2, 10)
algorithms = {
    'KMeans': KMeans(random_state=0),
    'Spectral': SpectralClustering(random_state=0, assign_labels='discretize'),
    'GMM': GaussianMixture(random_state=0)
}

# Compare clustering algorithms and assign labels
results = {}
base_labels = ["Low", "Moderate", "High", "Critical"]  # Basic criticality labels
for name, algorithm in algorithms.items():
    silhouettes = []
    for k in range_clusters:
        if name != 'GMM':  # Use 'n_clusters' for non-GMM algorithms
            algorithm.set_params(n_clusters=k)
        else:  # Use 'n_components' for GMM
            algorithm.set_params(n_components=k)
        labels = algorithm.fit_predict(principal_df)
        score = silhouette_score(principal_df, labels)
        silhouettes.append(score)
    optimal_k = range_clusters[np.argmax(silhouettes)]
    if name != 'GMM':
        algorithm.set_params(n_clusters=optimal_k)
    else:
        algorithm.set_params(n_components=optimal_k)
    final_labels = algorithm.fit_predict(principal_df)
    results[name] = {
        'optimal_k': optimal_k,
        'silhouette': max(silhouettes),
        'labels': final_labels
    }

# Assign criticality labels post hoc
for name, result in results.items():
    unique_labels = np.unique(result['labels'])
    n_labels = len(unique_labels)
    criticality_labels = {label: base_labels[i % len(base_labels)] for i, label in enumerate(unique_labels)}
    results[name]['label_names'] = criticality_labels

# Output results
for name, result in results.items():
    print(f"{name}: Optimal clusters = {result['optimal_k']}, Silhouette score = {result['silhouette']}")
    print("Cluster labels:", result['label_names'])

# Plot results
fig, ax = plt.subplots(1, len(results), figsize=(20, 5), sharey=True)
for i, (name, result) in enumerate(results.items()):
    unique_labels = np.unique(result['labels'])
    label_names = [result['label_names'][label] for label in unique_labels]
    sns.scatterplot(x='PC1', y='PC2', hue=[result['label_names'][label] for label in result['labels']], style=[result['label_names'][label] for label in result['labels']], data=principal_df, palette='viridis', s=100, alpha=0.8, ax=ax[i])
    ax[i].set_title(f'{name} Clustering (k={result["optimal_k"]})')
    ax[i].set_xlabel('Principal Component 1')
    ax[i].set_ylabel('Principal Component 2')
    ax[i].legend(title='Cluster', labels=label_names, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

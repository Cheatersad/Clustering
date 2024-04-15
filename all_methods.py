import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering, AffinityPropagation, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib.pyplot as plt

# Suppress the convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Generating sample data
X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)

# Initialize clustering algorithms
clustering_algorithms = [
    ("KMeans", KMeans(n_clusters=5)),
    ("Agglomerative", AgglomerativeClustering(n_clusters=5)),
    ("DBSCAN", DBSCAN(eps=0.5, min_samples=5)),
    ("MeanShift", MeanShift()),
    ("Spectral", SpectralClustering(n_clusters=5)),
    ("AffinityPropagation", AffinityPropagation()),
    ("OPTICS", OPTICS(min_samples=5)),
    ("Birch", Birch(n_clusters=5)),
    ("GMM", GaussianMixture(n_components=5))
]

# Create DataFrame to store results
df = pd.DataFrame(X, columns=["X1", "X2"])

# Perform clustering and append results to DataFrame
for name, algorithm in clustering_algorithms:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        algorithm.fit(X)
    
    if hasattr(algorithm, 'labels_'):
        df[name] = algorithm.labels_
    else:
        # For GMM, assign labels based on highest probability
        probabilities = algorithm.predict_proba(X)
        labels = np.argmax(probabilities, axis=1)
        df[name] = labels

# Save DataFrame to CSV
df.to_csv('clustering_results.csv', index=False)
print("Results saved to clustering_results.csv")

# Plot the clustering results
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i, (name, _) in enumerate(clustering_algorithms):
    ax = axes[i]
    ax.scatter(df['X1'], df['X2'], c=df[name], cmap='viridis')
    ax.set_title(name)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

plt.tight_layout()
plt.show()

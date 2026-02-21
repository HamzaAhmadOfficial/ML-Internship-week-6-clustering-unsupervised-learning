"""
Week 6 - Task 6.2
Hierarchical Clustering & Dendrograms

Features:
✔ Agglomerative Clustering
✔ Dendrogram visualization
✔ Multiple linkage comparison
✔ PCA cluster visualization
✔ Comparison with K-Means
✔ Silhouette score evaluation
"""


#  Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage


# Create outputs folder if it doesn't exist
outputs_folder = "outputs"
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)
    print(f"\nCreated '{outputs_folder}' folder for saving outputs...")


#  Load Dataset

print("\nLoading dataset...")

iris = load_iris()
X = iris.data
feature_names = iris.feature_names

df = pd.DataFrame(X, columns=feature_names)


#  Scale Dataset

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#  Create Dendrograms

print("\nCreating dendrograms...")

linkage_methods = ['ward', 'single', 'complete', 'average']

for method in linkage_methods:
    plt.figure(figsize=(8, 5))
    Z = linkage(X_scaled, method=method)
    dendrogram(Z)
    plt.title(f"Dendrogram ({method.capitalize()} Linkage)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.savefig(os.path.join(outputs_folder, f"dendrogram_{method}.png"))
    plt.show()


#  Choose Optimal Number of Clusters

# (From visual inspection — Iris usually best at 3)
optimal_clusters = 3
print(f"\nOptimal clusters chosen: {optimal_clusters}")


# Train Agglomerative Clustering

agglo = AgglomerativeClustering(
    n_clusters=optimal_clusters,
    linkage='ward'
)

agglo_labels = agglo.fit_predict(X_scaled)

df["Hierarchical_Cluster"] = agglo_labels


#  PCA Visualization

print("\nApplying PCA for visualization...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agglo_labels)
plt.title("Hierarchical Clustering (PCA 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig(os.path.join(outputs_folder, "hierarchical_clusters_pca.png"))
plt.show()


#  K-Means Comparison

print("\nRunning K-Means for comparison...")

kmeans = KMeans(
    n_clusters=optimal_clusters,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

df["KMeans_Cluster"] = kmeans_labels


#  Silhouette Score Comparison

hier_score = silhouette_score(X_scaled, agglo_labels)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)

print("\nSilhouette Scores Comparison")
print("--------------------------------")
print(f"Hierarchical Clustering Score: {hier_score:.4f}")
print(f"K-Means Score: {kmeans_score:.4f}")


#  PCA Side-by-Side Visualization

plt.figure(figsize=(10, 4))

# Hierarchical
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agglo_labels)
plt.title("Hierarchical Clustering")

# KMeans
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
plt.title("K-Means Clustering")

plt.savefig(os.path.join(outputs_folder, "hierarchical_vs_kmeans.png"))
plt.show()


#  Save Results

df.to_csv(os.path.join(outputs_folder, "hierarchical_clustering_results.csv"), index=False)

print(f"\nAll outputs saved to the '{outputs_folder}' folder:")
print(f"  - {outputs_folder}/dendrogram_ward.png")
print(f"  - {outputs_folder}/dendrogram_single.png")
print(f"  - {outputs_folder}/dendrogram_complete.png")
print(f"  - {outputs_folder}/dendrogram_average.png")
print(f"  - {outputs_folder}/hierarchical_clusters_pca.png")
print(f"  - {outputs_folder}/hierarchical_vs_kmeans.png")
print(f"  - {outputs_folder}/hierarchical_clustering_results.csv")
print("\n Hierarchical Clustering Task Completed Successfully!")
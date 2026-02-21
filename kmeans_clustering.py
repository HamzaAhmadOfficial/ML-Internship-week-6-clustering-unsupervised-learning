#  Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#  Load Dataset

print("\nLoading dataset...")

iris = load_iris()
X = iris.data
feature_names = iris.feature_names

df = pd.DataFrame(X, columns=feature_names)


#  Scale Data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#  Elbow Method

print("\nCalculating Elbow Method...")

inertia = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++',
                    random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.savefig("elbow_curve.png")
plt.show()


#  Silhouette Scores

print("\nCalculating Silhouette Scores...")

silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++',
                    random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Plot Silhouette Scores
plt.figure()
plt.plot(K_range, silhouette_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.savefig("silhouette_scores.png")
plt.show()

# Best K
best_k = K_range[np.argmax(silhouette_scores)]
print(f"\nBest K based on Silhouette Score: {best_k}")


#  Compare Initializations

print("\nComparing initialization methods...")

kmeans_random = KMeans(
    n_clusters=best_k,
    init='random',
    random_state=42,
    n_init=10
)

kmeans_plus = KMeans(
    n_clusters=best_k,
    init='k-means++',
    random_state=42,
    n_init=10
)

labels_random = kmeans_random.fit_predict(X_scaled)
labels_plus = kmeans_plus.fit_predict(X_scaled)

print("Random Init Inertia:", kmeans_random.inertia_)
print("K-Means++ Inertia:", kmeans_plus.inertia_)


#  PCA for 2D Visualization

print("\nApplying PCA for visualization...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_plus)
plt.title("K-Means Clusters (PCA 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig("kmeans_clusters_pca.png")
plt.show()


#  Create Cluster Profiles

print("\nCreating cluster profiles...")

df["Cluster"] = labels_plus

cluster_profile = df.groupby("Cluster").mean()

print("\nCluster Profile Statistics:")
print(cluster_profile)

cluster_profile.to_csv("cluster_profiles.csv")


#  Save Cluster Assignments

print("\nSaving clustered dataset...")

df.to_csv("kmeans_clustered_data.csv", index=False)

print("\n K-Means Clustering Task Completed Successfully!")
"""
Week 6 - Task 6.3
DBSCAN & Density-Based Clustering

Features:
DBSCAN clustering
eps & min_samples experimentation
Noise point detection
Arbitrary-shaped dataset
Visualization
K-Means comparison
Cluster statistics
"""


#  Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Create outputs folder if it doesn't exist
outputs_folder = "outputs"
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)
    print(f"\nCreated '{outputs_folder}' folder for saving outputs...")


#  Create Dataset (Non-Spherical)

print("\nGenerating dataset...")

X, _ = make_moons(n_samples=400, noise=0.05, random_state=42)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#  Experiment with Parameters

eps_values = [0.1, 0.3, 0.5, 1.0]
min_samples_values = [3, 5, 10]

results = []

print("\nTesting DBSCAN parameters...")

for eps in eps_values:
    for min_s in min_samples_values:

        dbscan = DBSCAN(eps=eps, min_samples=min_s)
        labels = dbscan.fit_predict(X_scaled)

        # Count clusters (excluding noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Silhouette only if valid clusters exist
        if n_clusters > 1:
            score = silhouette_score(X_scaled, labels)
        else:
            score = -1

        results.append((eps, min_s, n_clusters, score))

        print(
            f"eps={eps}, min_samples={min_s} -> "
            f"clusters={n_clusters}, silhouette={score:.3f}"
        )

# Convert results to DataFrame
results_df = pd.DataFrame(
    results,
    columns=["eps", "min_samples", "clusters", "silhouette_score"]
)

results_df.to_csv(os.path.join(outputs_folder, "dbscan_parameter_results.csv"), index=False)


#  Choose Best Parameters

best_row = results_df.loc[results_df["silhouette_score"].idxmax()]
best_eps = best_row["eps"]
best_min_samples = int(best_row["min_samples"])

print("\nBest Parameters Found:")
print(best_row)


#  Train Final DBSCAN Model

dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
db_labels = dbscan.fit_predict(X_scaled)

# Identify noise points
noise_points = db_labels == -1
print(f"\nNumber of noise points: {np.sum(noise_points)}")


#  Visualize DBSCAN Clusters

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=db_labels)
plt.scatter(
    X_scaled[noise_points, 0],
    X_scaled[noise_points, 1],
    marker="x",
    label="Noise"
)
plt.title("DBSCAN Clustering")
plt.legend()
plt.savefig(os.path.join(outputs_folder, "dbscan_clusters.png"))
plt.show()


#  Apply K-Means on Same Data

print("\nRunning K-Means for comparison...")

# Use same number of clusters detected by DBSCAN
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)

kmeans = KMeans(n_clusters=n_clusters_db, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)


#  Side-by-Side Comparison

plt.figure(figsize=(10, 4))

# DBSCAN
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=db_labels)
plt.title("DBSCAN")

# KMeans
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels)
plt.title("K-Means")

plt.savefig(os.path.join(outputs_folder, "dbscan_vs_kmeans.png"))
plt.show()


#  Silhouette Comparison

if n_clusters_db > 1:
    db_score = silhouette_score(X_scaled, db_labels)
else:
    db_score = -1

km_score = silhouette_score(X_scaled, kmeans_labels)

print("\nSilhouette Scores")
print("-------------------------")
print(f"DBSCAN Score : {db_score:.4f}")
print(f"K-Means Score: {km_score:.4f}")


#  Save Cluster Assignments

df = pd.DataFrame(X_scaled, columns=["Feature_1", "Feature_2"])
df["DBSCAN_Cluster"] = db_labels
df["KMeans_Cluster"] = kmeans_labels

df.to_csv(os.path.join(outputs_folder, "dbscan_clustering_results.csv"), index=False)


#  Documentation Output

print("\nWhen to Use DBSCAN vs K-Means:")
print("""
DBSCAN Works Better When:
- Clusters have irregular shapes
- Dataset contains noise/outliers
- Number of clusters is unknown

K-Means Works Better When:
- Clusters are spherical
- Dataset has low noise
- Fast computation is required
""")

print(f"\nAll outputs saved to the '{outputs_folder}' folder:")
print(f"  - {outputs_folder}/dbscan_parameter_results.csv")
print(f"  - {outputs_folder}/dbscan_clusters.png")
print(f"  - {outputs_folder}/dbscan_vs_kmeans.png")
print(f"  - {outputs_folder}/dbscan_clustering_results.csv")
print("\n DBSCAN Clustering Task Completed Successfully!")
"""
Week 6 - Task 6.4
Dimensionality Reduction (PCA & t-SNE)

Features:
- PCA dimensionality reduction
- Explained variance visualization
- PCA 2D & 3D visualization
- t-SNE visualization
- Classification comparison
- Accuracy & training time evaluation
"""


# 1. Import Libraries

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mpl_toolkits.mplot3d import Axes3D


# Create outputs folder if it doesn't exist
outputs_folder = "outputs"
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)
    print(f"\nCreated '{outputs_folder}' folder for saving outputs...")


#  Load High-Dimensional Dataset

print("\nLoading dataset...")

digits = load_digits()
X = digits.data
y = digits.target

print("Dataset shape:", X.shape)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#  Apply PCA (50 Components)

print("\nApplying PCA (50 components)...")

pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X_scaled)


#  Explained Variance Ratio Plot

plt.figure()
plt.plot(pca_50.explained_variance_ratio_)
plt.title("Explained Variance Ratio (PCA Components)")
plt.xlabel("Component")
plt.ylabel("Variance Ratio")
plt.savefig(os.path.join(outputs_folder, "pca_explained_variance.png"))
plt.show()


#  Cumulative Explained Variance

plt.figure()
plt.plot(np.cumsum(pca_50.explained_variance_ratio_))
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.savefig(os.path.join(outputs_folder, "pca_cumulative_variance.png"))
plt.show()


#  PCA 2D Visualization

print("\nCreating PCA 2D visualization...")

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y)
plt.title("PCA 2D Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(outputs_folder, "pca_2d.png"))
plt.show()


#  PCA 3D Visualization

print("\nCreating PCA 3D visualization...")

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y)
ax.set_title("PCA 3D Visualization")
plt.savefig(os.path.join(outputs_folder, "pca_3d.png"))
plt.show()


#  Apply t-SNE

print("\nApplying t-SNE (this may take time)...")

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.title("t-SNE Visualization")
plt.savefig(os.path.join(outputs_folder, "tsne_2d.png"))
plt.show()


#  Train Classifier Comparison

print("\nTraining classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---- Original Features ----
start = time.time()
model_original = LogisticRegression(max_iter=5000)
model_original.fit(X_train, y_train)
train_time_original = time.time() - start

pred_original = model_original.predict(X_test)
acc_original = accuracy_score(y_test, pred_original)

# ---- PCA Reduced Features ----
X_train_pca, X_test_pca, _, _ = train_test_split(
    X_pca_50, y, test_size=0.2, random_state=42
)

start = time.time()
model_pca = LogisticRegression(max_iter=5000)
model_pca.fit(X_train_pca, y_train)
train_time_pca = time.time() - start

pred_pca = model_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, pred_pca)


#  Compare Results

print("\nClassification Comparison")
print("-----------------------------------")
print(f"Original Accuracy : {acc_original:.4f}")
print(f"PCA Accuracy      : {acc_pca:.4f}")
print(f"Original Train Time: {train_time_original:.4f}s")
print(f"PCA Train Time     : {train_time_pca:.4f}s")

# Save comparison
results = pd.DataFrame({
    "Method": ["Original Features", "PCA Features"],
    "Accuracy": [acc_original, acc_pca],
    "Training Time": [train_time_original, train_time_pca]
})

results.to_csv(os.path.join(outputs_folder, "dimensionality_reduction_results.csv"), index=False)

print(f"\nAll outputs saved to the '{outputs_folder}' folder:")
print(f"  - {outputs_folder}/pca_explained_variance.png")
print(f"  - {outputs_folder}/pca_cumulative_variance.png")
print(f"  - {outputs_folder}/pca_2d.png")
print(f"  - {outputs_folder}/pca_3d.png")
print(f"  - {outputs_folder}/tsne_2d.png")
print(f"  - {outputs_folder}/dimensionality_reduction_results.csv")
print("\n Dimensionality Reduction Task Completed Successfully!")
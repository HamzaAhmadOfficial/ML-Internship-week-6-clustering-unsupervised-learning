#  Week 6 — Clustering & Unsupervised Learning

## Machine Learning Internship — Week 6

This repository contains implementations of clustering algorithms and dimensionality reduction techniques completed as part of my Machine Learning Internship. The objective of this week is to understand unsupervised learning, where machine learning models discover hidden patterns and structures in unlabeled datasets.

##  Topics Covered

- K-Means Clustering
- Hierarchical (Agglomerative) Clustering
- DBSCAN (Density-Based Clustering)
- Dimensionality Reduction
  - PCA (Principal Component Analysis)
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)

##  Repository Structure

unsupervised-learning-clustering-projects/
│
├── kmeans_clustering.py
├── hierarchical_clustering.py
├── dbscan_clustering.py
├── dimensionality_reduction.py
└── README.md

##  Installation & Setup

Clone the repository:

git clone https://github.com/HamzaAhmadOfficial/ML-Internship-week-6-clustering-unsupervised-learning.git


Install required libraries:

pip install numpy pandas matplotlib scikit-learn scipy

##  Task 6.1 — K-Means Clustering

K-Means clustering is implemented to group similar data points and determine the optimal number of clusters.

Features
- Elbow Method for optimal K selection
- Silhouette Score evaluation
- PCA-based 2D cluster visualization
- Comparison between random initialization and k-means++
- Cluster profiling statistics
- Saving cluster assignments

Output Files
- elbow_curve.png
- silhouette_scores.png
- kmeans_clusters_pca.png
- cluster_profiles.csv
- kmeans_clustered_data.csv

Run:
python kmeans_clustering.py

##  Task 6.2 — Hierarchical Clustering

Hierarchical clustering analyzes relationships between samples using dendrograms and linkage methods.

Features
- Agglomerative clustering
- Dendrogram visualization
- Linkage comparison (single, complete, average, ward)
- PCA visualization
- Comparison with K-Means
- Silhouette score evaluation

Output Files
- dendrogram_ward.png
- dendrogram_single.png
- dendrogram_complete.png
- dendrogram_average.png
- hierarchical_vs_kmeans.png

Run:
python hierarchical_clustering.py

##  Task 6.3 — DBSCAN Clustering

DBSCAN identifies clusters based on density and detects noise or outliers.

Features
- DBSCAN clustering implementation
- Parameter experimentation (eps and min_samples)
- Noise point identification
- Visualization of arbitrary-shaped clusters
- Comparison with K-Means
- Silhouette score comparison

Output Files
- dbscan_clusters.png
- dbscan_vs_kmeans.png
- dbscan_parameter_results.csv
- dbscan_clustering_results.csv

Run:
python dbscan_clustering.py

##  Task 6.4 — Dimensionality Reduction (PCA & t-SNE)

Dimensionality reduction techniques are applied to visualize and improve learning efficiency on high-dimensional datasets.

Dataset
- Scikit-learn Digits Dataset

Features
- PCA with 50 components
- Explained variance ratio visualization
- PCA 2D and 3D visualization
- t-SNE visualization
- Classification using original vs reduced features
- Accuracy and training time comparison

Output Files
- pca_explained_variance.png
- pca_cumulative_variance.png
- pca_2d.png
- pca_3d.png
- tsne_2d.png
- dimensionality_reduction_results.csv

Run:
python dimensionality_reduction.py

##  Algorithms Comparison Summary

Algorithm | Strengths | Weaknesses
---------|-----------|----------
K-Means | Fast and scalable | Assumes spherical clusters
Hierarchical | Shows cluster hierarchy | Computationally expensive
DBSCAN | Detects noise and irregular shapes | Sensitive to parameters
PCA | Efficient dimensionality reduction | Linear transformation
t-SNE | Excellent visualization | Computationally expensive

##  Key Learning Outcomes

- Understanding unsupervised learning workflows
- Selecting optimal cluster numbers
- Comparing clustering algorithms
- Handling noisy datasets
- Visualizing high-dimensional data
- Evaluating models using silhouette score
- Improving performance using dimensionality reduction

##  Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

##  Author

Hamza Ahmad
BS Data Science Student | Machine Learning Intern

GitHub: https://github.com/HamzaAhmadOfficial

##  Future Improvements

- Add interactive visualizations
- Apply clustering on real-world datasets
- Deploy dashboard using Streamlit
- Compare additional algorithms (OPTICS, UMAP)

##  License

This project is created for educational and internship learning purposes.

##  Week 6 Successfully Completed — Clustering & Unsupervised Learning
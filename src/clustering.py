# src/clustering.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score,
                              davies_bouldin_score,
                              calinski_harabasz_score)
import yaml
import os

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_optimal_k(X_pca, config):
    """
    Test different values of K and find the best one.
    Uses 3 metrics:
    - Inertia (elbow method): lower is better
    - Silhouette score: higher is better (-1 to 1)
    - Davies-Bouldin: lower is better

    Args:
        X_pca: PCA-reduced data
        config: project config
    Returns:
        best_k: optimal number of clusters
        scores: dict of scores for each K
    """
    k_min = config['clustering']['k_range_min']
    k_max = config['clustering']['k_range_max']
    K_range = range(k_min, k_max + 1)

    print(f"Finding optimal K (testing K={k_min} to K={k_max})...")

    inertia = []
    silhouette = []
    davies_bouldin = []
    calinski = []

    for k in K_range:
        print(f"  Testing K={k}...", end=' ')
        km = KMeans(
            n_clusters=k,
            random_state=config['clustering']['random_state'],
            n_init=config['clustering']['n_init']
        )
        labels = km.fit_predict(X_pca)

        inertia.append(km.inertia_)
        sil = silhouette_score(X_pca, labels)
        db = davies_bouldin_score(X_pca, labels)
        ch = calinski_harabasz_score(X_pca, labels)

        silhouette.append(sil)
        davies_bouldin.append(db)
        calinski.append(ch)

        print(f"Silhouette={sil:.3f}, Davies-Bouldin={db:.3f}")

    # Best K = highest silhouette score
    best_k = list(K_range)[np.argmax(silhouette)]
    print(f"\n✅ Best K = {best_k} (highest silhouette score: {max(silhouette):.3f})")

    scores = {
        'k_range': list(K_range),
        'inertia': inertia,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski': calinski
    }

    return best_k, scores

def plot_k_selection(scores, save_dir='results/'):
    """Plot elbow curve and silhouette scores to visualize K selection"""
    K_range = scores['k_range']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Elbow curve
    axes[0].plot(K_range, scores['inertia'], marker='o', color='coral', linewidth=2)
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True, alpha=0.3)

    # Silhouette score
    best_idx = np.argmax(scores['silhouette'])
    axes[1].plot(K_range, scores['silhouette'], marker='o', color='steelblue', linewidth=2)
    axes[1].axvline(x=K_range[best_idx], color='red', linestyle='--',
                    label=f'Best K={K_range[best_idx]}')
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score (higher=better)")
    axes[1].set_title("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Davies-Bouldin
    axes[2].plot(K_range, scores['davies_bouldin'], marker='o', color='green', linewidth=2)
    axes[2].set_xlabel("Number of Clusters (K)")
    axes[2].set_ylabel("Davies-Bouldin (lower=better)")
    axes[2].set_title("Davies-Bouldin Score")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/06_k_selection.png', dpi=150)
    plt.show()
    print("K selection plots saved!")

def run_kmeans(X_pca, best_k, config):
    """
    Run K-Means clustering with the optimal K.
    K-Means groups cells so that cells in the same group
    are as similar as possible.

    Returns: cluster labels, fitted kmeans object
    """
    print(f"Running K-Means with K={best_k}...")

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=config['clustering']['random_state'],
        n_init=config['clustering']['n_init']
    )
    labels = kmeans.fit_predict(X_pca)

    # Count cells per cluster
    unique, counts = np.unique(labels, return_counts=True)
    print("  Cluster sizes:")
    for cluster, count in zip(unique, counts):
        print(f"    Subtype {cluster}: {count} cells ({count/len(labels)*100:.1f}%)")

    return labels, kmeans

def run_hierarchical(X_pca, best_k):
    """
    Run Hierarchical clustering as a comparison.
    Builds a tree by merging most similar cells step by step.

    Returns: cluster labels
    """
    print(f"Running Hierarchical clustering with K={best_k}...")
    agglo = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    labels = agglo.fit_predict(X_pca)
    print("  Hierarchical clustering complete ✓")
    return labels

def evaluate_clusters(X_pca, labels, method_name):
    """
    Calculate 3 cluster quality metrics.

    Returns: dict of scores
    """
    sil = silhouette_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)

    print(f"\n{method_name} Evaluation:")
    print(f"  Silhouette Score:        {sil:.3f}  (↑ higher better, max=1)")
    print(f"  Davies-Bouldin Score:    {db:.3f}  (↓ lower better, min=0)")
    print(f"  Calinski-Harabasz Score: {ch:.1f} (↑ higher better)")

    return {'silhouette': sil, 'davies_bouldin': db, 'calinski': ch}

def plot_clusters_umap(X_umap, labels, title, save_path):
    """Plot UMAP with cluster colors"""
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
              '#ff7f00', '#a65628', '#f781bf', '#999999']

    n_clusters = len(np.unique(labels))
    plt.figure(figsize=(12, 9))

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        plt.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            label=f'Subtype {cluster_id} (n={mask.sum()})',
            color=colors[cluster_id % len(colors)],
            s=15, alpha=0.7, edgecolors='none'
        )

    plt.title(title, fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Molecular Subtype", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {save_path}")

def run_clustering(X_pca, X_umap, config_path='config.yaml'):
    """
    MAIN FUNCTION — runs complete clustering pipeline.

    Returns:
        kmeans_labels: cluster assignments from K-Means
        best_k: optimal number of clusters
        kmeans: fitted K-Means model
    """
    print("=" * 50)
    print("STARTING CLUSTERING PIPELINE")
    print("=" * 50)

    config = load_config(config_path)
    os.makedirs('results', exist_ok=True)

    # Find optimal K
    best_k, scores = find_optimal_k(X_pca, config)
    plot_k_selection(scores, save_dir='results/')

    # Run K-Means
    kmeans_labels, kmeans = run_kmeans(X_pca, best_k, config)

    # Run Hierarchical for comparison
    hier_labels = run_hierarchical(X_pca, best_k)

    # Evaluate both
    km_scores = evaluate_clusters(X_pca, kmeans_labels, "K-Means")
    hier_scores = evaluate_clusters(X_pca, hier_labels, "Hierarchical")

    # Plot results
    plot_clusters_umap(X_umap, kmeans_labels,
                       f"K-Means Clustering (K={best_k})",
                       'results/07_kmeans_clusters.png')

    plot_clusters_umap(X_umap, hier_labels,
                       f"Hierarchical Clustering (K={best_k})",
                       'results/08_hierarchical_clusters.png')

    # Save cluster labels
    np.save('results/kmeans_labels.npy', kmeans_labels)
    np.save('results/hier_labels.npy', hier_labels)

    print("\n" + "=" * 50)
    print("CLUSTERING COMPLETE")
    print(f"Best number of subtypes: {best_k}")
    print("=" * 50)

    return kmeans_labels, best_k, kmeans
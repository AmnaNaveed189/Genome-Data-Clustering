# src/dimensionality.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import yaml
import os

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pca(X_scaled, config):
    """
    Reduce dimensions from thousands of genes to 50 components.
    PCA finds the directions of maximum variance in the data.

    Think of it as: compressing a 5000-page book to its 50 key themes.

    Args:
        X_scaled: preprocessed data (cells x genes)
        config: project config dict
    Returns:
        X_pca: reduced data (cells x 50)
        pca: fitted PCA object (save for new patients)
    """
    n_components = config['pca']['n_components']
    print(f"Running PCA: {X_scaled.shape[1]} genes → {n_components} components...")

    pca = PCA(
        n_components=n_components,
        random_state=config['pca']['random_state']
    )
    X_pca = pca.fit_transform(X_scaled)

    # How much information did we keep?
    total_variance = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance explained by {n_components} PCs: {total_variance:.1f}%")
    print(f"  Shape after PCA: {X_pca.shape}")

    return X_pca, pca

def plot_pca_variance(pca, save_dir='results/'):
    """
    Plot how much variance each principal component explains.
    This helps decide how many components to keep.
    """
    explained = pca.explained_variance_ratio_
    cumulative = explained.cumsum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance per component
    ax1.bar(range(1, len(explained) + 1), explained * 100,
            color='steelblue', alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Variance per Component")
    ax1.set_xlim(0, len(explained) + 1)

    # Cumulative variance
    ax2.plot(range(1, len(cumulative) + 1), cumulative,
             marker='o', markersize=3, color='coral', linewidth=2)
    ax2.axhline(y=80, color='green', linestyle='--', label='80% threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', label='90% threshold')
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/04_pca_variance.png', dpi=150)
    plt.show()
    print(f"PCA variance plot saved!")

def run_umap(X_pca, config):
    """
    Reduce from 50 PCA components to 2D for visualization.
    UMAP preserves neighborhood structure — similar cells stay close.

    Think of it as: creating a 2D map where similar patients
    are placed near each other.

    Args:
        X_pca: PCA-reduced data (cells x 50)
        config: project config dict
    Returns:
        X_umap: 2D coordinates (cells x 2)
        reducer: fitted UMAP object
    """
    print(f"Running UMAP: {X_pca.shape[1]} components → 2D...")
    print("  (This may take 2-5 minutes for 4645 cells — please wait)")

    reducer = umap.UMAP(
        n_components=config['umap']['n_components'],
        n_neighbors=config['umap']['n_neighbors'],
        min_dist=config['umap']['min_dist'],
        random_state=config['umap']['random_state'],
        verbose=True
    )
    X_umap = reducer.fit_transform(X_pca)

    print(f"  UMAP complete! Shape: {X_umap.shape}")

    return X_umap, reducer

def plot_umap_basic(X_umap, save_dir='results/'):
    """Plot basic UMAP without cluster colors (just the landscape)"""
    plt.figure(figsize=(10, 8))
    plt.scatter(X_umap[:, 0], X_umap[:, 1],
                alpha=0.5, s=15, color='steelblue', edgecolors='none')
    plt.title("UMAP Projection of Melanoma Cells", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(f'{save_dir}/05_umap_basic.png', dpi=150)
    plt.show()
    print("Basic UMAP plot saved!")

def run_dimensionality_reduction(X_scaled, config_path='config.yaml'):
    """
    MAIN FUNCTION — runs complete dimensionality reduction.

    Returns:
        X_pca: PCA-reduced data
        X_umap: 2D UMAP coordinates
        pca: fitted PCA object
        reducer: fitted UMAP object
    """
    print("=" * 50)
    print("STARTING DIMENSIONALITY REDUCTION")
    print("=" * 50)

    config = load_config(config_path)

    # Run PCA
    X_pca, pca = run_pca(X_scaled, config)
    plot_pca_variance(pca, save_dir='results/')

    # Run UMAP
    X_umap, reducer = run_umap(X_pca, config)
    plot_umap_basic(X_umap, save_dir='results/')

    # Save UMAP coordinates for later
    os.makedirs('results', exist_ok=True)
    np.save('results/X_pca.npy', X_pca)
    np.save('results/X_umap.npy', X_umap)
    print("Coordinates saved to results/")

    print("\n" + "=" * 50)
    print("DIMENSIONALITY REDUCTION COMPLETE")
    print("=" * 50)

    return X_pca, X_umap, pca, reducer
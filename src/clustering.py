import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _normalize_higher_better(values):
    arr = np.asarray(values, dtype=float)
    if arr.max() == arr.min():
        return np.ones_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def _normalize_lower_better(values):
    arr = np.asarray(values, dtype=float)
    if arr.max() == arr.min():
        return np.ones_like(arr)
    return (arr.max() - arr) / (arr.max() - arr.min())


def _compute_inertia_improvement(inertia):
    inertia = np.asarray(inertia, dtype=float)
    gains = np.zeros_like(inertia)
    for i in range(1, len(inertia)):
        prev = inertia[i - 1]
        curr = inertia[i]
        gains[i] = ((prev - curr) / prev) if prev else 0.0
    return gains


def find_optimal_k(X_pca, config):
    """
    Test different K values and select best K using combined evidence.

    Returns:
        best_k: selected K
        scores: dict with metric lists and selection table
    """
    k_min = config['clustering']['k_range_min']
    k_max = config['clustering']['k_range_max']
    k_range = list(range(k_min, k_max + 1))

    print(f"Finding optimal K (testing K={k_min} to K={k_max})...")

    rows = []
    for k in k_range:
        print(f"  Testing K={k}...", end=' ')
        km = KMeans(
            n_clusters=k,
            random_state=config['clustering']['random_state'],
            n_init=config['clustering']['n_init'],
        )
        labels = km.fit_predict(X_pca)

        sil = silhouette_score(X_pca, labels)
        db = davies_bouldin_score(X_pca, labels)
        ch = calinski_harabasz_score(X_pca, labels)

        rows.append(
            {
                'k': k,
                'inertia': float(km.inertia_),
                'silhouette': float(sil),
                'davies_bouldin': float(db),
                'calinski': float(ch),
            }
        )
        print(
            f"Silhouette={sil:.3f}, Davies-Bouldin={db:.3f}, "
            f"Calinski-Harabasz={ch:.1f}"
        )

    report_df = pd.DataFrame(rows)
    report_df['inertia_gain'] = _compute_inertia_improvement(report_df['inertia'])

    report_df['silhouette_norm'] = _normalize_higher_better(report_df['silhouette'])
    report_df['davies_bouldin_norm'] = _normalize_lower_better(report_df['davies_bouldin'])
    report_df['calinski_norm'] = _normalize_higher_better(report_df['calinski'])
    report_df['inertia_gain_norm'] = _normalize_higher_better(report_df['inertia_gain'])

    report_df['combined_score'] = (
        0.35 * report_df['silhouette_norm']
        + 0.25 * report_df['davies_bouldin_norm']
        + 0.25 * report_df['calinski_norm']
        + 0.15 * report_df['inertia_gain_norm']
    )

    best_idx = int(report_df['combined_score'].idxmax())
    best_k = int(report_df.loc[best_idx, 'k'])

    print(
        f"\n✅ Selected K = {best_k} "
        f"(combined evidence score: {report_df.loc[best_idx, 'combined_score']:.3f})"
    )

    scores = {
        'k_range': report_df['k'].tolist(),
        'inertia': report_df['inertia'].tolist(),
        'silhouette': report_df['silhouette'].tolist(),
        'davies_bouldin': report_df['davies_bouldin'].tolist(),
        'calinski': report_df['calinski'].tolist(),
        'inertia_gain': report_df['inertia_gain'].tolist(),
        'combined_score': report_df['combined_score'].tolist(),
        'selection_report': report_df,
    }

    return best_k, scores


def write_k_selection_report(scores, best_k, save_dir='results/'):
    os.makedirs(save_dir, exist_ok=True)
    report_df = scores['selection_report'].copy()
    report_df['selected'] = report_df['k'] == best_k
    report_df.to_csv(f'{save_dir}/k_selection_report.csv', index=False)

    selected_row = report_df[report_df['selected']].iloc[0]
    selection_summary = {
        'selected_k': int(best_k),
        'k2_candidate_metrics': (
            report_df[report_df['k'] == 2]
            .to_dict(orient='records')[0]
            if (report_df['k'] == 2).any()
            else None
        ),
        'selection_basis': {
            'method': 'weighted-multi-metric',
            'weights': {
                'silhouette': 0.35,
                'davies_bouldin': 0.25,
                'calinski_harabasz': 0.25,
                'inertia_gain': 0.15,
            },
            'selected_metrics': {
                'silhouette': float(selected_row['silhouette']),
                'davies_bouldin': float(selected_row['davies_bouldin']),
                'calinski_harabasz': float(selected_row['calinski']),
                'inertia_gain': float(selected_row['inertia_gain']),
                'combined_score': float(selected_row['combined_score']),
            },
        },
    }

    with open(f'{save_dir}/k_selection_summary.json', 'w', encoding='utf-8') as f:
        json.dump(selection_summary, f, indent=2)

    print(f"Selection report saved: {save_dir}/k_selection_report.csv")


def analyze_kmeans_stability(X_pca, best_k, config, save_dir='results/'):
    """Run repeated K-Means with different seeds and summarize stability."""
    os.makedirs(save_dir, exist_ok=True)

    base_seed = int(config['clustering'].get('random_state', 42))
    n_runs = int(config['clustering'].get('stability_runs', 5))
    n_runs = max(3, n_runs)

    labels_runs = []
    seed_rows = []

    print(f"Assessing clustering stability over {n_runs} runs...")
    for i in range(n_runs):
        seed = base_seed + i
        km = KMeans(
            n_clusters=best_k,
            random_state=seed,
            n_init=config['clustering']['n_init'],
        )
        labels = km.fit_predict(X_pca)
        labels_runs.append(labels)
        seed_rows.append({'run': i + 1, 'seed': seed, 'inertia': float(km.inertia_)})

    ari_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            ari_scores.append(adjusted_rand_score(labels_runs[i], labels_runs[j]))

    mean_ari = float(np.mean(ari_scores)) if ari_scores else 1.0
    min_ari = float(np.min(ari_scores)) if ari_scores else 1.0
    max_ari = float(np.max(ari_scores)) if ari_scores else 1.0

    status = 'stable' if mean_ari >= 0.90 else 'exploratory'

    pd.DataFrame(seed_rows).to_csv(f'{save_dir}/kmeans_stability_runs.csv', index=False)
    stability_summary = {
        'selected_k': int(best_k),
        'n_runs': int(n_runs),
        'pairwise_ari': {
            'mean': mean_ari,
            'min': min_ari,
            'max': max_ari,
        },
        'stability_status': status,
    }
    with open(f'{save_dir}/kmeans_stability_summary.json', 'w', encoding='utf-8') as f:
        json.dump(stability_summary, f, indent=2)

    print(f"Stability summary saved: {save_dir}/kmeans_stability_summary.json")
    return stability_summary


def plot_k_selection(scores, save_dir='results/'):
    """Plot K selection metrics."""
    k_range = scores['k_range']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(k_range, scores['inertia'], marker='o', color='coral', linewidth=2)
    axes[0, 0].set_xlabel('Number of Clusters (K)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].grid(True, alpha=0.3)

    best_idx = int(np.argmax(scores['combined_score']))
    axes[0, 1].plot(k_range, scores['silhouette'], marker='o', color='steelblue', linewidth=2)
    axes[0, 1].axvline(
        x=k_range[best_idx],
        color='red',
        linestyle='--',
        label=f'Selected K={k_range[best_idx]}',
    )
    axes[0, 1].set_xlabel('Number of Clusters (K)')
    axes[0, 1].set_ylabel('Silhouette Score (higher=better)')
    axes[0, 1].set_title('Silhouette Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(k_range, scores['davies_bouldin'], marker='o', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Number of Clusters (K)')
    axes[1, 0].set_ylabel('Davies-Bouldin (lower=better)')
    axes[1, 0].set_title('Davies-Bouldin Score')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(k_range, scores['combined_score'], marker='o', color='purple', linewidth=2)
    axes[1, 1].axvline(x=k_range[best_idx], color='red', linestyle='--')
    axes[1, 1].set_xlabel('Number of Clusters (K)')
    axes[1, 1].set_ylabel('Combined Evidence Score')
    axes[1, 1].set_title('Combined K Selection Score')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/06_k_selection.png', dpi=150)
    plt.show()
    print('K selection plots saved!')


def run_kmeans(X_pca, best_k, config):
    """
    Run K-Means clustering with the selected K.

    Returns: cluster labels, fitted kmeans object
    """
    print(f'Running K-Means with K={best_k}...')

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=config['clustering']['random_state'],
        n_init=config['clustering']['n_init'],
    )
    labels = kmeans.fit_predict(X_pca)

    unique, counts = np.unique(labels, return_counts=True)
    print('  Cluster sizes:')
    for cluster, count in zip(unique, counts):
        print(f'    Subtype {cluster}: {count} cells ({count/len(labels)*100:.1f}%)')

    return labels, kmeans


def run_hierarchical(X_pca, best_k):
    """
    Run Hierarchical clustering as comparison.

    Returns: cluster labels
    """
    print(f'Running Hierarchical clustering with K={best_k}...')
    agglo = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    labels = agglo.fit_predict(X_pca)
    print('  Hierarchical clustering complete ✓')
    return labels


def evaluate_clusters(X_pca, labels, method_name):
    """
    Calculate cluster quality metrics.

    Returns: dict of scores
    """
    sil = silhouette_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)

    print(f'\n{method_name} Evaluation:')
    print(f'  Silhouette Score:        {sil:.3f}  (↑ higher better, max=1)')
    print(f'  Davies-Bouldin Score:    {db:.3f}  (↓ lower better, min=0)')
    print(f'  Calinski-Harabasz Score: {ch:.1f} (↑ higher better)')

    return {'silhouette': sil, 'davies_bouldin': db, 'calinski': ch}


def plot_clusters_umap(X_umap, labels, title, save_path):
    """Plot UMAP with cluster colors."""
    colors = [
        '#e41a1c',
        '#377eb8',
        '#4daf4a',
        '#984ea3',
        '#ff7f00',
        '#a65628',
        '#f781bf',
        '#999999',
    ]

    n_clusters = len(np.unique(labels))
    plt.figure(figsize=(12, 9))

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        plt.scatter(
            X_umap[mask, 0],
            X_umap[mask, 1],
            label=f'Subtype {cluster_id} (n={mask.sum()})',
            color=colors[cluster_id % len(colors)],
            s=15,
            alpha=0.7,
            edgecolors='none',
        )

    plt.title(title, fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Molecular Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Plot saved: {save_path}')


def run_clustering(X_pca, X_umap, config_path='config.yaml'):
    """
    MAIN FUNCTION — runs complete clustering pipeline.

    Returns:
        kmeans_labels, best_k, kmeans, hier_labels, metadata
    """
    print('=' * 50)
    print('STARTING CLUSTERING PIPELINE')
    print('=' * 50)

    config = load_config(config_path)
    os.makedirs('results', exist_ok=True)

    best_k, scores = find_optimal_k(X_pca, config)
    plot_k_selection(scores, save_dir='results/')
    write_k_selection_report(scores, best_k, save_dir='results/')

    kmeans_labels, kmeans = run_kmeans(X_pca, best_k, config)
    hier_labels = run_hierarchical(X_pca, best_k)

    km_scores = evaluate_clusters(X_pca, kmeans_labels, 'K-Means')
    hier_scores = evaluate_clusters(X_pca, hier_labels, 'Hierarchical')

    stability_summary = analyze_kmeans_stability(X_pca, best_k, config, save_dir='results/')

    plot_clusters_umap(
        X_umap,
        kmeans_labels,
        f'K-Means Clustering (K={best_k})',
        'results/07_kmeans_clusters.png',
    )

    plot_clusters_umap(
        X_umap,
        hier_labels,
        f'Hierarchical Clustering (K={best_k})',
        'results/08_hierarchical_clusters.png',
    )

    np.save('results/kmeans_labels.npy', kmeans_labels)
    np.save('results/hier_labels.npy', hier_labels)

    clustering_meta = {
        'selected_k': int(best_k),
        'kmeans_scores': km_scores,
        'hierarchical_scores': hier_scores,
        'stability': stability_summary,
    }
    with open('results/clustering_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(clustering_meta, f, indent=2)

    print('\n' + '=' * 50)
    print('CLUSTERING COMPLETE')
    print(f'Best number of subtypes: {best_k}')
    print('=' * 50)

    return kmeans_labels, best_k, kmeans, hier_labels, clustering_meta

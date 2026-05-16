import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


def find_top_genes_per_cluster(expr_df, labels, top_n=20):
    """
    Find genes that are most different (differentially expressed)
    between clusters.

    Args:
        expr_df: expression DataFrame (cells x genes)
        labels: cluster labels
        top_n: how many top genes to return per cluster
    Returns:
        dict of {cluster_id: top_genes_dataframe}
    """
    print('Finding differentially expressed genes per cluster...')
    n_clusters = len(np.unique(labels))
    results = {}

    for cluster_id in range(n_clusters):
        print(f'  Analyzing Subtype {cluster_id}...')

        in_cluster = expr_df[labels == cluster_id]
        out_cluster = expr_df[labels != cluster_id]

        gene_stats = []
        for gene in expr_df.columns:
            t_stat, p_val = ttest_ind(
                in_cluster[gene].values,
                out_cluster[gene].values,
                equal_var=False,
            )
            mean_in = in_cluster[gene].mean()
            mean_out = out_cluster[gene].mean()
            fold_change = mean_in - mean_out

            gene_stats.append(
                {
                    'cluster': int(cluster_id),
                    'gene': gene,
                    'p_value': float(p_val),
                    'fold_change': float(fold_change),
                    'mean_in_cluster': float(mean_in),
                    'mean_out_cluster': float(mean_out),
                }
            )

        stats_df = pd.DataFrame(gene_stats)
        stats_df['abs_fold_change'] = stats_df['fold_change'].abs()
        stats_df = stats_df.sort_values(['p_value', 'abs_fold_change'], ascending=[True, False]).head(top_n)

        results[cluster_id] = stats_df
        print(
            f"    Top gene: {stats_df.iloc[0]['gene']} "
            f"(p={stats_df.iloc[0]['p_value']:.2e})"
        )

    return results


def save_top_genes_table(top_genes_per_cluster, save_dir='results/'):
    os.makedirs(save_dir, exist_ok=True)
    combined = pd.concat(top_genes_per_cluster.values(), ignore_index=True)
    combined.to_csv(f'{save_dir}/top_genes_per_cluster.csv', index=False)
    print(f'Top genes table saved: {save_dir}/top_genes_per_cluster.csv')
    return combined


def _confidence_from_markers(cluster_df):
    strong = ((cluster_df['p_value'] < 0.01) & (cluster_df['abs_fold_change'] >= 0.5)).sum()
    if strong >= 5:
        return 'high'
    if strong >= 2:
        return 'medium'
    return 'low'


def build_cluster_marker_summary(top_genes_per_cluster, save_dir='results/'):
    """Build evidence-backed marker summary per cluster with confidence labels."""
    os.makedirs(save_dir, exist_ok=True)

    summary = []
    for cluster_id, df in top_genes_per_cluster.items():
        top_markers = df.sort_values('p_value').head(5)['gene'].tolist()
        avg_effect = float(df['abs_fold_change'].head(10).mean())
        confidence = _confidence_from_markers(df)

        if avg_effect >= 1.0:
            pattern = 'strong differential expression pattern'
        elif avg_effect >= 0.5:
            pattern = 'moderate differential expression pattern'
        else:
            pattern = 'subtle differential expression pattern'

        summary.append(
            {
                'cluster_id': int(cluster_id),
                'marker_genes': top_markers,
                'summary': f'Cluster shows {pattern} relative to other clusters.',
                'confidence': confidence,
                'evidence': {
                    'mean_abs_fold_change_top10': avg_effect,
                    'min_p_value': float(df['p_value'].min()),
                },
            }
        )

    with open(f'{save_dir}/cluster_marker_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'clusters': summary}, f, indent=2)

    print(f'Marker summary saved: {save_dir}/cluster_marker_summary.json')
    return summary


def plot_expression_heatmap(expr_df, labels, top_genes_per_cluster, save_dir='results/'):
    """
    Create heatmap showing gene expression patterns across clusters.
    """
    print('Creating expression heatmap...')

    all_top_genes = []
    for _, genes_df in top_genes_per_cluster.items():
        all_top_genes.extend(genes_df['gene'].tolist()[:10])
    all_top_genes = list(dict.fromkeys(all_top_genes))

    sort_idx = np.argsort(labels)
    sorted_expr = expr_df.iloc[sort_idx][all_top_genes]

    plt.figure(figsize=(16, 10))
    sns.heatmap(
        sorted_expr.T,
        cmap='RdBu_r',
        xticklabels=False,
        yticklabels=True,
        center=0,
        cbar_kws={'label': 'Expression Level (z-score)'},
    )
    plt.title('Gene Expression Heatmap by Molecular Subtype', fontsize=14)
    plt.xlabel(f'Cells (sorted by subtype) — n={len(labels)}')
    plt.ylabel('Top Differentially Expressed Genes')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/09_expression_heatmap.png', dpi=150)
    plt.show()
    print('Heatmap saved!')


def plot_top_genes_barplot(top_genes_per_cluster, save_dir='results/'):
    """Bar plot of most significant genes per cluster"""
    n_clusters = len(top_genes_per_cluster)
    fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 6))

    if n_clusters == 1:
        axes = [axes]

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

    for cluster_id, genes_df in top_genes_per_cluster.items():
        top10 = genes_df.head(10)
        axes[cluster_id].barh(
            top10['gene'],
            -np.log10(top10['p_value'] + 1e-300),
            color=colors[cluster_id % len(colors)],
            alpha=0.8,
        )
        axes[cluster_id].set_title(f'Subtype {cluster_id}\nTop Genes')
        axes[cluster_id].set_xlabel('-log10(p-value)')
        axes[cluster_id].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/10_top_genes.png', dpi=150)
    plt.show()
    print('Top genes plot saved!')

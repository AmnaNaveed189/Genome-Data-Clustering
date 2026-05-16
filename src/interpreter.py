# src/interpreter.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os

def find_top_genes_per_cluster(expr_df, labels, top_n=20):
    """
    Find genes that are most different (differentially expressed)
    between clusters. These genes DEFINE each subtype.

    Think of it as: what makes Subtype 0 cells different
    from Subtype 1 cells at the gene level?

    Args:
        expr_df: expression DataFrame (cells x genes)
        labels: cluster labels
        top_n: how many top genes to return per cluster
    Returns:
        dict of {cluster_id: top_genes_dataframe}
    """
    print("Finding differentially expressed genes per cluster...")
    n_clusters = len(np.unique(labels))
    results = {}

    for cluster_id in range(n_clusters):
        print(f"  Analyzing Subtype {cluster_id}...")

        # This cluster vs all others
        in_cluster = expr_df[labels == cluster_id]
        out_cluster = expr_df[labels != cluster_id]

        gene_stats = []
        for gene in expr_df.columns:
            t_stat, p_val = ttest_ind(
                in_cluster[gene].values,
                out_cluster[gene].values,
                equal_var=False
            )
            mean_in = in_cluster[gene].mean()
            mean_out = out_cluster[gene].mean()
            fold_change = mean_in - mean_out  # log fold change

            gene_stats.append({
                'gene': gene,
                'p_value': p_val,
                'fold_change': fold_change,
                'mean_in_cluster': mean_in,
                'mean_out_cluster': mean_out
            })

        stats_df = pd.DataFrame(gene_stats)
        stats_df['abs_fold_change'] = stats_df['fold_change'].abs()
        stats_df = stats_df.sort_values('p_value').head(top_n)

        results[cluster_id] = stats_df
        print(f"    Top gene: {stats_df.iloc[0]['gene']} "
              f"(p={stats_df.iloc[0]['p_value']:.2e})")

    return results

def plot_expression_heatmap(expr_df, labels, top_genes_per_cluster, save_dir='results/'):
    """
    Create heatmap showing gene expression patterns across clusters.
    Each row = a gene, each column = a cell.
    Color = expression level (red=high, blue=low)
    """
    print("Creating expression heatmap...")

    # Collect top genes from all clusters
    all_top_genes = []
    for cluster_id, genes_df in top_genes_per_cluster.items():
        all_top_genes.extend(genes_df['gene'].tolist()[:10])
    all_top_genes = list(dict.fromkeys(all_top_genes))  # remove duplicates

    # Sort cells by cluster
    sort_idx = np.argsort(labels)
    sorted_expr = expr_df.iloc[sort_idx][all_top_genes]
    sorted_labels = labels[sort_idx]

    plt.figure(figsize=(16, 10))
    sns.heatmap(
        sorted_expr.T,
        cmap='RdBu_r',
        xticklabels=False,
        yticklabels=True,
        center=0,
        cbar_kws={'label': 'Expression Level (z-score)'}
    )
    plt.title("Gene Expression Heatmap by Molecular Subtype", fontsize=14)
    plt.xlabel(f"Cells (sorted by subtype) — n={len(labels)}")
    plt.ylabel("Top Differentially Expressed Genes")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/09_expression_heatmap.png', dpi=150)
    plt.show()
    print("Heatmap saved!")

def plot_top_genes_barplot(top_genes_per_cluster, save_dir='results/'):
    """Bar plot of most significant genes per cluster"""
    n_clusters = len(top_genes_per_cluster)
    fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 6))

    if n_clusters == 1:
        axes = [axes]

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
              '#ff7f00', '#a65628', '#f781bf', '#999999']

    for cluster_id, genes_df in top_genes_per_cluster.items():
        top10 = genes_df.head(10)
        axes[cluster_id].barh(
            top10['gene'],
            -np.log10(top10['p_value'] + 1e-300),
            color=colors[cluster_id % len(colors)],
            alpha=0.8
        )
        axes[cluster_id].set_title(f"Subtype {cluster_id}\nTop Genes")
        axes[cluster_id].set_xlabel("-log10(p-value)")
        axes[cluster_id].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/10_top_genes.png', dpi=150)
    plt.show()
    print("Top genes plot saved!")
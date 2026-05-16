# src/run_pipeline.py
# Run this to execute the ENTIRE pipeline end to end

import json
import os
import sys

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import run_clustering
from src.dimensionality import run_dimensionality_reduction
from src.interpreter import (
    build_cluster_marker_summary,
    find_top_genes_per_cluster,
    plot_expression_heatmap,
    plot_top_genes_barplot,
    save_top_genes_table,
)
from src.predictor import save_all_models, train_subtype_predictor
from src.preprocessor import run_preprocessing


def _write_feature_metadata(gene_names, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        'expected_input_length': len(gene_names),
        'feature_names': gene_names,
    }
    with open(os.path.join(save_dir, 'feature_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _write_cluster_summary(labels, marker_summary, clustering_meta, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    subtype_counts = pd.Series(labels).value_counts().sort_index()
    stability = clustering_meta.get('stability', {})
    k_status = stability.get('stability_status', 'exploratory')

    summary_payload = {
        'dataset': 'GSE72056 — Melanoma scRNA-seq',
        'total_cells': int(len(labels)),
        'total_clusters': int(len(subtype_counts)),
        'selection_status': k_status,
        'selection_note': (
            'Final K accepted as stable based on repeated seeded runs.'
            if k_status == 'stable'
            else 'Current K is exploratory; stability did not meet the acceptance threshold.'
        ),
        'evidence': {
            'k_selection_summary_path': 'results/k_selection_summary.json',
            'k_stability_summary_path': 'results/kmeans_stability_summary.json',
        },
        'subtypes': [
            {
                'subtype_id': int(cluster_id),
                'cell_count': int(count),
                'cell_percent': float((count / len(labels)) * 100.0),
                'marker_genes': next(
                    (
                        item['marker_genes']
                        for item in marker_summary
                        if int(item['cluster_id']) == int(cluster_id)
                    ),
                    [],
                ),
                'summary': next(
                    (
                        item['summary']
                        for item in marker_summary
                        if int(item['cluster_id']) == int(cluster_id)
                    ),
                    'Marker summary unavailable.',
                ),
                'confidence': next(
                    (
                        item['confidence']
                        for item in marker_summary
                        if int(item['cluster_id']) == int(cluster_id)
                    ),
                    'low',
                ),
            }
            for cluster_id, count in subtype_counts.items()
        ],
    }

    with open(os.path.join(save_dir, 'cluster_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, indent=2)



def main():
    print('\n' + '🧬' * 25)
    print('GENOMIC CLUSTERING PIPELINE — FULL RUN')
    print('🧬' * 25 + '\n')

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # PHASE 1: Preprocessing
    X_scaled, scaler, gene_names, cell_names, _ = run_preprocessing('config.yaml')

    # PHASE 2: Dimensionality Reduction
    X_pca, X_umap, pca, reducer = run_dimensionality_reduction(X_scaled, 'config.yaml')

    # PHASE 3: Clustering
    labels, best_k, kmeans, hier_labels, clustering_meta = run_clustering(X_pca, X_umap, 'config.yaml')

    # PHASE 4: Interpretation
    with open('config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    expr_df = pd.read_csv(config['data']['processed_expression'], index_col=0)
    top_genes = find_top_genes_per_cluster(expr_df, labels)
    top_genes_df = save_top_genes_table(top_genes, save_dir='results/')
    marker_summary = build_cluster_marker_summary(top_genes, save_dir='results/')
    plot_expression_heatmap(expr_df, labels, top_genes)
    plot_top_genes_barplot(top_genes)

    # PHASE 5: Train predictor + save models
    clf = train_subtype_predictor(X_pca, labels, config)
    save_all_models(scaler, pca, kmeans, clf)
    _write_feature_metadata(gene_names, save_dir='models')

    # Save final results for dashboard
    results_df = pd.DataFrame(
        {
            'cell_index': list(range(len(cell_names))),
            'cell_id': cell_names,
            'kmeans_subtype': labels,
            'hier_subtype': hier_labels,
            'final_subtype': labels,
            'umap_1': X_umap[:, 0],
            'umap_2': X_umap[:, 1],
        }
    )
    results_df.to_csv('results/cluster_assignments.csv', index=False)

    # Keep legacy-compatible artifact
    results_df[['cell_id', 'final_subtype', 'umap_1', 'umap_2']].rename(
        columns={'final_subtype': 'subtype'}
    ).to_csv('results/final_cluster_assignments.csv', index=False)

    _write_cluster_summary(labels, marker_summary, clustering_meta, save_dir='results')

    final_report = {
        'selected_k': int(best_k),
        'stability': clustering_meta.get('stability', {}),
        'k_selection_report': 'results/k_selection_report.csv',
        'marker_summary_report': 'results/cluster_marker_summary.json',
        'top_genes_report': 'results/top_genes_per_cluster.csv',
        'limitations': [
            'Subtype labels are discovered clusters, not externally validated clinical labels.',
            'Interpretation confidence depends on differential-expression strength and sample composition.',
            'Final K should be treated as exploratory when stability status is exploratory.',
        ],
    }
    with open('results/final_report.json', 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2)

    print('\n' + '✅' * 25)
    print('PIPELINE COMPLETE!')
    print('✅' * 25)
    print('\nOutputs:')
    print('  results/  — plots, cluster assignments, and evidence reports')
    print('  models/   — saved ML models and feature metadata')
    print('\nNext steps:')
    print('  Run API:       uvicorn api.app:app --reload')
    print('  Run Dashboard: streamlit run dashboard/app.py')


if __name__ == '__main__':
    main()

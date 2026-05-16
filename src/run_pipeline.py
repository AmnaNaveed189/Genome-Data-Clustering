# src/run_pipeline.py
# Run this to execute the ENTIRE pipeline end to end

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

from src.preprocessor import run_preprocessing
from src.dimensionality import run_dimensionality_reduction
from src.clustering import run_clustering
from src.interpreter import (find_top_genes_per_cluster,
                               plot_expression_heatmap,
                               plot_top_genes_barplot)
from src.predictor import train_subtype_predictor, save_all_models

def main():
    print("\n" + "🧬" * 25)
    print("GENOMIC CLUSTERING PIPELINE — FULL RUN")
    print("🧬" * 25 + "\n")

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # PHASE 1: Preprocessing
    X_scaled, scaler, gene_names, cell_names, metadata = \
        run_preprocessing('config.yaml')

    # PHASE 2: Dimensionality Reduction
    X_pca, X_umap, pca, reducer = \
        run_dimensionality_reduction(X_scaled, 'config.yaml')

    # PHASE 3: Clustering
    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    from src.clustering import run_clustering
    labels, best_k, kmeans = run_clustering(X_pca, X_umap, 'config.yaml')

    # PHASE 4: Interpretation
    expr_df = pd.read_csv(config['data']['processed_expression'], index_col=0)
    top_genes = find_top_genes_per_cluster(expr_df, labels)
    plot_expression_heatmap(expr_df, labels, top_genes)
    plot_top_genes_barplot(top_genes)

    # PHASE 5: Train predictor + save models
    clf = train_subtype_predictor(X_pca, labels, config)
    save_all_models(scaler, pca, kmeans, clf)

    # Save final results
    results_df = pd.DataFrame({
        'cell_id': cell_names,
        'subtype': labels,
        'umap_1': X_umap[:, 0],
        'umap_2': X_umap[:, 1]
    })
    results_df.to_csv('results/final_cluster_assignments.csv', index=False)

    print("\n" + "✅" * 25)
    print("PIPELINE COMPLETE!")
    print("✅" * 25)
    print("\nOutputs:")
    print("  results/  — all plots + cluster assignments")
    print("  models/   — saved ML models")
    print("\nNext steps:")
    print("  Run API:       uvicorn api.app:app --reload")
    print("  Run Dashboard: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()

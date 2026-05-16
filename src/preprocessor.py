# src/preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
import os

def load_config(config_path='config.yaml'):
    """Load project configuration"""
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['_config_dir'] = os.path.dirname(config_path)
    return config

def resolve_path(config, path_value):
    """Resolve a config path relative to the config file location."""
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(config['_config_dir'], path_value))

def load_data(config):
    """
    Load raw expression and metadata CSV files
    Returns: expression DataFrame, metadata DataFrame
    """
    print("Loading raw data...")

    expr_path = resolve_path(config, config['data']['raw_expression'])
    meta_path = resolve_path(config, config['data']['raw_metadata'])

    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path, index_col=0)

    print(f"  Expression loaded: {expr.shape[0]} genes x {expr.shape[1]} cells")
    print(f"  Metadata loaded:   {meta.shape}")

    return expr, meta

def separate_metadata(expr):
    """
    The GSE72056 dataset has metadata in first 3 rows.
    This function separates them from gene expression rows.

    Returns: clean expression DataFrame, metadata DataFrame
    """
    print("Separating metadata from expression data...")

    # First 3 rows are: tumor ID, malignant status, cell type
    # Everything after row 3 is gene expression
    metadata_rows = ['tumor', 'malignant(1=yes,2=no,0=unresolved)',
                     'non-malignant cell type (1=T,2=B,3=Macro.,'
                     '4=Endo.,5=CAF,6=NK)']

    # Separate based on index
    meta = expr.iloc[:3, :]
    genes = expr.iloc[3:, :]

    print(f"  Metadata rows: {meta.shape[0]}")
    print(f"  Gene rows: {genes.shape[0]}")

    return genes, meta

def filter_low_variance_genes(expr, percentile=20):
    """
    Remove genes that barely change across cells.
    These carry no useful information for clustering.

    Args:
        expr: gene expression DataFrame (genes x cells)
        percentile: remove genes below this variance percentile
    Returns: filtered DataFrame
    """
    print(f"Filtering low-variance genes (removing bottom {percentile}%)...")

    gene_variance = expr.var(axis=1)
    threshold = gene_variance.quantile(percentile / 100)
    expr_filtered = expr[gene_variance > threshold]

    print(f"  Before: {expr.shape[0]} genes")
    print(f"  After:  {expr_filtered.shape[0]} genes")
    print(f"  Removed: {expr.shape[0] - expr_filtered.shape[0]} genes")

    return expr_filtered

def keep_top_variable_genes(expr, n_genes=5000):
    """
    Keep only the most variable genes.
    This speeds up computation significantly.

    Args:
        expr: gene expression DataFrame
        n_genes: number of top variable genes to keep
    Returns: filtered DataFrame
    """
    print(f"Keeping top {n_genes} most variable genes...")

    gene_variance = expr.var(axis=1)
    top_genes = gene_variance.nlargest(n_genes).index
    expr_top = expr.loc[top_genes]

    print(f"  Final gene count: {expr_top.shape[0]}")

    return expr_top

def handle_missing_values(expr):
    """
    Fill missing values with gene mean expression.
    """
    missing = expr.isnull().sum().sum()
    if missing > 0:
        print(f"Filling {missing} missing values with gene means...")
        # expr is cells x genes at call site, so column means = gene means.
        expr = expr.fillna(expr.mean(axis=0))
    else:
        print("No missing values found ✓")
    return expr

def log_transform(expr):
    """
    Apply log1p transformation to normalize skewed distribution.
    log1p(x) = log(x + 1) — the +1 prevents log(0) errors.
    """
    print("Applying log1p transformation...")
    expr_log = np.log1p(expr.astype(float))
    print("  Log transformation complete ✓")
    return expr_log

def standardize(X):
    """
    Standardize data: mean=0, std=1 for each gene.
    Prevents high-expression genes from dominating.

    Returns: scaled numpy array, fitted scaler object
    """
    print("Standardizing data (mean=0, std=1)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Mean: {X_scaled.mean():.4f} (should be ~0)")
    print(f"  Std:  {X_scaled.std():.4f} (should be ~1)")
    return X_scaled, scaler

def run_preprocessing(config_path='config.yaml'):
    """
    MAIN FUNCTION — runs complete preprocessing pipeline.
    Call this to preprocess your data.

    Returns:
        X_scaled: numpy array ready for ML (cells x genes)
        scaler: fitted StandardScaler
        gene_names: list of gene names
        cell_names: list of cell names
        metadata: DataFrame with cell labels
    """
    print("=" * 50)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 50)

    config = load_config(config_path)

    # Step 1: Load data
    expr, _ = load_data(config)

    # Step 2: Separate metadata from gene rows
    expr_genes, metadata = separate_metadata(expr)

    # Step 3: Transpose → rows=cells, columns=genes
    expr_T = expr_genes.T
    print(f"\nTransposed shape: {expr_T.shape} (cells x genes)")

    # Step 4: Handle missing values
    expr_T = handle_missing_values(expr_T)

    # Step 5: Filter low-variance genes
    expr_filtered = filter_low_variance_genes(
        expr_T.T,  # back to genes x cells for variance calc
        percentile=config['preprocessing']['variance_percentile']
    ).T  # transpose back to cells x genes

    # Step 6: Keep top variable genes only
    expr_top = keep_top_variable_genes(
        expr_filtered.T,
        n_genes=config['preprocessing']['max_genes']
    ).T

    # Step 7: Log transform
    if config['preprocessing']['log_transform']:
        expr_log = log_transform(expr_top)
    else:
        expr_log = expr_top

    # Step 8: Standardize
    if config['preprocessing']['scale']:
        X_scaled, scaler = standardize(expr_log.values)
    else:
        X_scaled = expr_log.values
        scaler = None

    # Save processed data
    processed_expression_path = resolve_path(
        config, config['data']['processed_expression']
    )
    processed_metadata_path = resolve_path(
        config, config['data']['processed_metadata']
    )

    os.makedirs(os.path.dirname(processed_expression_path), exist_ok=True)
    expr_log.to_csv(processed_expression_path)
    metadata.T.to_csv(processed_metadata_path)
    print(f"\nProcessed data saved to {processed_expression_path}")

    gene_names = expr_top.columns.tolist()
    cell_names = expr_top.index.tolist()

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print(f"Final shape: {X_scaled.shape} (cells x genes)")
    print("=" * 50)

    return X_scaled, scaler, gene_names, cell_names, metadata.T

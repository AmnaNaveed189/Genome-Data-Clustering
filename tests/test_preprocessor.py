import numpy as np
import pandas as pd

from src.preprocessor import handle_missing_values, keep_top_variable_genes, standardize


def test_handle_missing_values_and_standardize():
    df = pd.DataFrame(
        {
            'g1': [1.0, np.nan, 3.0],
            'g2': [2.0, 2.0, 2.0],
            'g3': [0.0, 1.0, 2.0],
        }
    )
    filled = handle_missing_values(df)
    assert not filled.isna().any().any()

    X_scaled, scaler = standardize(filled.values)
    assert X_scaled.shape == filled.values.shape
    assert scaler is not None


def test_keep_top_variable_genes_returns_requested_count():
    expr = pd.DataFrame(
        {
            'c1': [1, 1, 1, 1],
            'c2': [2, 1, 4, 1],
            'c3': [3, 1, 7, 1],
        },
        index=['gene_a', 'gene_b', 'gene_c', 'gene_d'],
    )
    top = keep_top_variable_genes(expr, n_genes=2)
    assert top.shape[0] == 2

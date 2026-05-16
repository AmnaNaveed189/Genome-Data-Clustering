import numpy as np

from src.dimensionality import run_pca


def test_run_pca_shape_and_variance():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 20))
    config = {'pca': {'n_components': 5, 'random_state': 42}}

    X_pca, pca = run_pca(X, config)
    assert X_pca.shape == (60, 5)
    assert pca.explained_variance_ratio_.sum() > 0

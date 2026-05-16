import numpy as np
from sklearn.datasets import make_blobs

from src.clustering import analyze_kmeans_stability, find_optimal_k, write_k_selection_report


def test_find_optimal_k_and_reports(tmp_path):
    X, _ = make_blobs(n_samples=180, centers=3, n_features=6, random_state=42, cluster_std=0.8)
    config = {
        'clustering': {
            'k_range_min': 2,
            'k_range_max': 5,
            'n_init': 10,
            'random_state': 42,
            'stability_runs': 4,
        }
    }

    best_k, scores = find_optimal_k(X, config)
    assert 2 <= best_k <= 5
    assert 'selection_report' in scores
    assert 'combined_score' in scores

    write_k_selection_report(scores, best_k, save_dir=str(tmp_path))
    assert (tmp_path / 'k_selection_report.csv').exists()
    assert (tmp_path / 'k_selection_summary.json').exists()

    stability = analyze_kmeans_stability(X, best_k, config, save_dir=str(tmp_path))
    assert stability['selected_k'] == best_k
    assert 'stability_status' in stability
    assert (tmp_path / 'kmeans_stability_summary.json').exists()

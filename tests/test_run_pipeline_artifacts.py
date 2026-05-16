import json
import os

import numpy as np

from src.run_pipeline import _write_cluster_summary, _write_feature_metadata


def test_feature_metadata_contains_expected_dimensions(tmp_path):
    genes = [f'G{i}' for i in range(12)]
    _write_feature_metadata(genes, save_dir=str(tmp_path))

    path = tmp_path / 'feature_metadata.json'
    assert path.exists()
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['expected_input_length'] == 12
    assert len(payload['feature_names']) == 12


def test_cluster_summary_artifact_written(tmp_path):
    labels = np.array([0, 1, 1, 0, 2, 2])
    marker_summary = [
        {'cluster_id': 0, 'marker_genes': ['A'], 'summary': 's0', 'confidence': 'high'},
        {'cluster_id': 1, 'marker_genes': ['B'], 'summary': 's1', 'confidence': 'medium'},
        {'cluster_id': 2, 'marker_genes': ['C'], 'summary': 's2', 'confidence': 'low'},
    ]
    clustering_meta = {'stability': {'stability_status': 'stable'}}

    _write_cluster_summary(labels, marker_summary, clustering_meta, save_dir=str(tmp_path))

    summary_path = tmp_path / 'cluster_summary.json'
    assert os.path.exists(summary_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['total_cells'] == 6
    assert summary['total_clusters'] == 3
    assert len(summary['subtypes']) == 3

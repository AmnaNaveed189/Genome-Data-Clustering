import importlib
import json

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def _prepare_artifacts(models_dir, results_dir, n_features=8):
    rng = np.random.default_rng(7)
    X = np.abs(rng.normal(size=(200, n_features)))
    y = rng.integers(0, 2, size=200)

    scaler = StandardScaler().fit(np.log1p(X))
    X_scaled = scaler.transform(np.log1p(X))
    pca = PCA(n_components=4, random_state=42).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    clf = RandomForestClassifier(n_estimators=20, random_state=42).fit(X_pca, y)

    joblib.dump(scaler, models_dir / 'scaler.pkl')
    joblib.dump(pca, models_dir / 'pca.pkl')
    joblib.dump(clf, models_dir / 'classifier.pkl')

    with open(models_dir / 'feature_metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'expected_input_length': n_features}, f)

    summary = {
        'dataset': 'test',
        'total_cells': 200,
        'total_clusters': 2,
        'selection_status': 'stable',
        'selection_note': 'test note',
        'evidence': {},
        'subtypes': [
            {
                'subtype_id': 0,
                'cell_count': 100,
                'cell_percent': 50.0,
                'marker_genes': ['G1'],
                'summary': 'Cluster summary test',
                'confidence': 'medium',
            }
        ],
    }
    with open(results_dir / 'cluster_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f)


def test_api_endpoints(tmp_path, monkeypatch):
    models_dir = tmp_path / 'models'
    results_dir = tmp_path / 'results'
    models_dir.mkdir()
    results_dir.mkdir()
    _prepare_artifacts(models_dir, results_dir)

    monkeypatch.setenv('MODELS_DIR', str(models_dir))
    monkeypatch.setenv('RESULTS_DIR', str(results_dir))

    import api.app as api_app

    importlib.reload(api_app)
    client = TestClient(api_app.app)

    health = client.get('/health')
    assert health.status_code == 200
    assert health.json()['models_loaded'] is True

    valid_payload = {'patient_id': 'p1', 'gene_expression': [1.0] * 8}
    pred = client.post('/predict-subtype', json=valid_payload)
    assert pred.status_code == 200
    body = pred.json()
    assert 'predicted_subtype' in body
    assert 'all_probabilities' in body

    invalid_payload = {'patient_id': 'p2', 'gene_expression': [1.0] * 7}
    bad = client.post('/predict-subtype', json=invalid_payload)
    assert bad.status_code == 422

    summary = client.get('/cluster-summary')
    assert summary.status_code == 200
    assert summary.json()['total_clusters'] == 2

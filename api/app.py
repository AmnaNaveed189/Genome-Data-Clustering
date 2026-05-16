import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title='🧬 Genomic Subtype Predictor API',
    description='Predict discovered molecular subtypes from gene expression data',
    version='1.1.0',
)

MODELS_DIR = os.getenv('MODELS_DIR', 'models')
RESULTS_DIR = os.getenv('RESULTS_DIR', 'results')


class PatientData(BaseModel):
    gene_expression: List[float]
    patient_id: str = 'unknown'


@lru_cache(maxsize=1)
def _load_models():
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    pca_path = os.path.join(MODELS_DIR, 'pca.pkl')
    clf_path = os.path.join(MODELS_DIR, 'classifier.pkl')

    if not (os.path.exists(scaler_path) and os.path.exists(pca_path) and os.path.exists(clf_path)):
        raise FileNotFoundError('Required model artifacts are missing. Run pipeline first.')

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    clf = joblib.load(clf_path)
    return scaler, pca, clf


@lru_cache(maxsize=1)
def _load_feature_metadata() -> Dict[str, object]:
    path = os.path.join(MODELS_DIR, 'feature_metadata.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_cluster_summary_artifact() -> Optional[Dict[str, object]]:
    path = os.path.join(RESULTS_DIR, 'cluster_summary.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _expected_input_length(scaler, metadata: Dict[str, object]) -> int:
    meta_len = metadata.get('expected_input_length')
    scaler_len = int(getattr(scaler, 'n_features_in_', 0))

    if meta_len is not None and scaler_len and int(meta_len) != scaler_len:
        raise HTTPException(
            status_code=500,
            detail='Model metadata mismatch: feature metadata does not match scaler features.',
        )

    if meta_len is not None:
        return int(meta_len)
    if scaler_len:
        return scaler_len

    raise HTTPException(status_code=500, detail='Cannot determine expected feature length from artifacts.')


@app.get('/')
def root():
    return {
        'message': 'Genomic Clustering API is running',
        'docs': '/docs',
        'health': '/health',
    }


@app.get('/health')
def health():
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    pca_path = os.path.join(MODELS_DIR, 'pca.pkl')
    clf_path = os.path.join(MODELS_DIR, 'classifier.pkl')

    models_loaded = all(os.path.exists(p) for p in [scaler_path, pca_path, clf_path])
    cluster_summary_available = os.path.exists(os.path.join(RESULTS_DIR, 'cluster_summary.json'))

    return {
        'status': 'healthy',
        'models_loaded': models_loaded,
        'cluster_summary_available': cluster_summary_available,
    }


@app.post('/predict-subtype')
def predict_subtype(patient: PatientData):
    """
    Predict discovered subtype for a new sample.
    Validates feature length, numeric types, and finite/non-negative values.
    """
    try:
        scaler, pca, clf = _load_models()
        metadata = _load_feature_metadata()
        expected_len = _expected_input_length(scaler, metadata)

        x = np.asarray(patient.gene_expression, dtype=float)
        if x.ndim != 1:
            raise HTTPException(status_code=422, detail='gene_expression must be a 1D list of numeric values.')
        if len(x) != expected_len:
            raise HTTPException(
                status_code=422,
                detail=f'Invalid feature length: expected {expected_len}, got {len(x)}.',
            )
        if not np.isfinite(x).all():
            raise HTTPException(status_code=422, detail='gene_expression contains NaN or infinite values.')
        if (x < 0).any():
            raise HTTPException(status_code=422, detail='gene_expression contains negative values.')

        x = np.log1p(x)
        x = scaler.transform([x])
        x = pca.transform(x)

        subtype = int(clf.predict(x)[0])
        probabilities = clf.predict_proba(x)[0]
        class_labels = [int(c) for c in clf.classes_]
        confidence = float(probabilities.max())

        marker_summary = _load_cluster_summary_artifact()
        subtype_note = 'Evidence-backed cluster label from trained model artifacts.'
        if marker_summary and isinstance(marker_summary.get('subtypes'), list):
            matched = [s for s in marker_summary['subtypes'] if int(s.get('subtype_id', -1)) == subtype]
            if matched:
                subtype_note = matched[0].get('summary', subtype_note)

        return {
            'patient_id': patient.patient_id,
            'predicted_subtype': subtype,
            'confidence_percent': round(confidence * 100, 2),
            'all_probabilities': {
                f'subtype_{label}': round(float(prob) * 100, 2)
                for label, prob in zip(class_labels, probabilities)
            },
            'cluster_note': subtype_note,
        }

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')


@app.get('/cluster-summary')
def cluster_summary():
    """Return discovered-cluster summary from generated artifacts."""
    artifact = _load_cluster_summary_artifact()
    if artifact:
        return artifact

    fallback = {
        'dataset': 'unknown',
        'total_cells': None,
        'total_clusters': None,
        'selection_status': 'exploratory',
        'selection_note': 'No cluster summary artifact found. Run the full pipeline to generate evidence reports.',
        'evidence': {
            'k_selection_summary_path': os.path.join(RESULTS_DIR, 'k_selection_summary.json'),
            'k_stability_summary_path': os.path.join(RESULTS_DIR, 'kmeans_stability_summary.json'),
        },
        'subtypes': [],
    }

    kmeans_path = os.path.join(MODELS_DIR, 'kmeans.pkl')
    if os.path.exists(kmeans_path):
        kmeans = joblib.load(kmeans_path)
        fallback['total_clusters'] = int(getattr(kmeans, 'n_clusters', 0)) or None

    labels_path = os.path.join(RESULTS_DIR, 'kmeans_labels.npy')
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        fallback['total_cells'] = int(len(labels))
        unique, counts = np.unique(labels, return_counts=True)
        fallback['subtypes'] = [
            {
                'subtype_id': int(k),
                'cell_count': int(v),
                'cell_percent': round(float((v / len(labels)) * 100.0), 4),
                'marker_genes': [],
                'summary': 'Evidence summary unavailable until interpretation artifacts are generated.',
                'confidence': 'low',
            }
            for k, v in zip(unique, counts)
        ]

    return fallback

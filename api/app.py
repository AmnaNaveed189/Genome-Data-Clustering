# api/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os

app = FastAPI(
    title="🧬 Genomic Subtype Predictor API",
    description="Predict molecular subtypes from gene expression data",
    version="1.0.0"
)

# Load models once at startup
MODELS_DIR = os.getenv("MODELS_DIR", "models/")

@app.get("/")
def root():
    return {
        "message": "Genomic Clustering API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": os.path.exists(MODELS_DIR)}

class PatientData(BaseModel):
    gene_expression: List[float]
    patient_id: str = "unknown"

@app.post("/predict-subtype")
def predict_subtype(patient: PatientData):
    """
    Predict molecular subtype for a new patient.
    Input: list of gene expression values
    Output: predicted subtype + confidence score
    """
    try:
        scaler = joblib.load(f'{MODELS_DIR}/scaler.pkl')
        pca    = joblib.load(f'{MODELS_DIR}/pca.pkl')
        clf    = joblib.load(f'{MODELS_DIR}/classifier.pkl')

        x = np.log1p(np.array(patient.gene_expression))
        x = scaler.transform([x])
        x = pca.transform(x)

        subtype = int(clf.predict(x)[0])
        probabilities = clf.predict_proba(x)[0]
        confidence = float(probabilities.max())

        subtype_info = {
            0: "High malignancy — aggressive treatment recommended",
            1: "Immune-active — immunotherapy may be effective",
            2: "Stromal-rich — targeted therapy indicated",
            3: "Mixed phenotype — further profiling recommended"
        }

        return {
            "patient_id": patient.patient_id,
            "predicted_subtype": subtype,
            "confidence_percent": round(confidence * 100, 2),
            "all_probabilities": {
                f"subtype_{i}": round(float(p) * 100, 2)
                for i, p in enumerate(probabilities)
            },
            "clinical_note": subtype_info.get(subtype, "Consult specialist")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cluster-summary")
def cluster_summary():
    """Return summary of discovered clusters"""
    return {
        "total_clusters": 4,
        "dataset": "GSE72056 — Melanoma scRNA-seq",
        "total_cells": 4645,
        "subtypes": {
            "Subtype 0": {"description": "Malignant cells", "marker_genes": ["MITF", "AXL"]},
            "Subtype 1": {"description": "T cells", "marker_genes": ["CD3D", "CD8A"]},
            "Subtype 2": {"description": "Macrophages", "marker_genes": ["CD68", "CSF1R"]},
            "Subtype 3": {"description": "Endothelial cells", "marker_genes": ["PECAM1"]}
        }
    }
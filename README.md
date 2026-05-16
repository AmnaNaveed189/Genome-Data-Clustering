# Genome Data Clustering

End-to-end machine learning workflow for discovering molecular subtypes from melanoma single-cell gene-expression data and deploying subtype prediction through an API and dashboard.

## What this project does

1. Preprocesses raw expression data.
2. Reduces dimensionality with PCA + UMAP.
3. Selects cluster count (K) using combined evidence (silhouette, Davies-Bouldin, Calinski-Harabasz, inertia trend).
4. Runs clustering and stability checks across repeated seeds.
5. Builds marker-gene interpretation summaries per discovered cluster.
6. Trains a classifier to predict discovered subtype labels.
7. Serves predictions and cluster summaries through FastAPI and Streamlit.

## Repository structure

```text
Genome-Data-Clustering/
├── api/
│   └── app.py
├── dashboard/
│   └── app.py
├── notebook/
├── results/
├── src/
│   ├── preprocessor.py
│   ├── dimensionality.py
│   ├── clustering.py
│   ├── interpreter.py
│   ├── predictor.py
│   └── run_pipeline.py
├── config.yaml
├── requirements.txt
└── environment.yml
```

## Reproducible run

### 1) Environment setup

```bash
cd /home/runner/work/Genome-Data-Clustering/Genome-Data-Clustering
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run full pipeline (one command)

```bash
python src/run_pipeline.py
```

### 3) Run API

```bash
uvicorn api.app:app --reload
```

- API docs: `http://127.0.0.1:8000/docs`

### 4) Run dashboard

```bash
streamlit run dashboard/app.py
```

## Demo flow (input → output → interpretation)

1. **Input**: raw expression + metadata paths defined in `config.yaml`.
2. **Discovery output**:
   - `results/cluster_assignments.csv`
   - `results/k_selection_report.csv`
   - `results/k_selection_summary.json`
   - `results/kmeans_stability_summary.json`
3. **Interpretation output**:
   - `results/top_genes_per_cluster.csv`
   - `results/cluster_marker_summary.json`
   - `results/cluster_summary.json`
4. **Prediction output**:
   - `models/scaler.pkl`, `models/pca.pkl`, `models/kmeans.pkl`, `models/classifier.pkl`
   - `models/feature_metadata.json`

## API behavior (current)

- `GET /health`: service status + artifact availability.
- `POST /predict-subtype`: validates feature length/types/finite values/non-negative values, then returns predicted subtype with class probabilities.
- `GET /cluster-summary`: returns artifact-backed discovered-cluster summary (`results/cluster_summary.json`) or a safe fallback when artifacts are missing.

## Client-facing deliverables

After a complete run, hand off:

1. `results/cluster_assignments.csv`
2. `results/k_selection_report.csv`
3. `results/k_selection_summary.json`
4. `results/kmeans_stability_summary.json`
5. `results/top_genes_per_cluster.csv`
6. `results/cluster_marker_summary.json`
7. `results/cluster_summary.json`
8. `results/final_report.json`
9. `models/` artifacts for deployment

## Current limitations

- Discovered subtype IDs are **unsupervised clusters**, not externally validated clinical labels.
- Biological interpretation confidence is based on differential-expression strength in the current dataset and can shift with cohort composition.
- If stability score indicates `exploratory`, selected K should be treated as provisional rather than final clinical evidence.
- Pipeline quality depends on input schema alignment and feature ordering matching training-time processing.

## Acceptance criteria for completion

- Pipeline runs end-to-end without manual intervention.
- API summary matches generated artifacts (no hardcoded subtype narrative/count mismatch).
- Tests pass for core modules and API endpoints.
- Final report includes K rationale, stability status, interpretation summary, and limitations.

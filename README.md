<<<<<<< HEAD
# Genome-Data-Clustering
=======
# Genome Data Clustering

Genome Data Clustering is an end-to-end machine learning project for discovering molecular subtypes from melanoma single-cell gene expression data and turning those subtypes into deployable prediction tools. The repository combines exploratory notebooks, reusable Python pipeline modules, a FastAPI service, and a Streamlit dashboard in one workflow.

The project starts with raw gene expression data, performs preprocessing and dimensionality reduction, clusters cells into subtype groups, interprets those clusters biologically through differential expression analysis, and then trains a classifier that can assign new samples to the discovered subtypes.

## Project Goals

- Discover meaningful structure in high-dimensional genomic data using unsupervised learning.
- Visualize subtype separation in lower-dimensional space.
- Identify genes that distinguish each discovered cluster.
- Convert unsupervised cluster labels into a supervised prediction model for future samples.
- Expose the trained system through an API, dashboard, and notebook environment.

## Problem the Project Solves

Single-cell gene expression datasets contain thousands of genes measured across thousands of cells. That makes direct interpretation difficult because:

- the data is high-dimensional,
- many genes carry little signal,
- biologically similar cells may still be noisy,
- and subtype discovery usually requires multiple processing and modeling stages.

This repository addresses that by using a staged workflow:

1. Clean and standardize gene expression data.
2. Reduce dimensionality while preserving major variance and local structure.
3. Cluster cells into candidate molecular subtypes.
4. Interpret the clusters with differential expression analysis.
5. Train a predictive model so new patient-like inputs can be assigned to a discovered subtype.

## Dataset Context

The code and comments indicate the project is built around the `GSE72056` melanoma single-cell dataset. The preprocessing logic assumes the raw expression file contains metadata rows at the top and gene expression rows below them.

Expected data layout from the repo:

- Raw expression: `data/raw/melanoma_expression.csv`
- Raw metadata: `data/raw/melanoma_metadata.csv`
- Processed expression: `data/processed/expression_clean.csv`
- Processed metadata: `data/processed/metadata_clean.csv`

## End-to-End Workflow

The core pipeline is orchestrated by [src/run_pipeline.py](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/src/run_pipeline.py:1).

Pipeline stages:

1. Preprocessing
   Filters genes, handles missing values, log-transforms expression values, and standardizes features.
2. Dimensionality reduction
   Uses PCA to compress the feature space and UMAP to create a 2D embedding for visualization.
3. Clustering
   Uses K-Means as the main subtype discovery model and hierarchical clustering for comparison.
4. Interpretation
   Finds top differentiating genes per cluster and generates biological interpretation plots.
5. Prediction
   Trains a Random Forest classifier on the discovered cluster labels.
6. Deployment
   Serves results through FastAPI and Streamlit.

## Main Technologies Used

### Core Language and Environment

- Python 3.10
- Conda environment via [environment.yml](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/environment.yml:1)
- Pip dependencies via [requirements.txt](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/requirements.txt:1)

### Data and Numerical Computing

- `pandas` for tabular gene expression and metadata handling
- `numpy` for vectorized numeric operations
- `scipy` for statistics, especially differential expression testing

### Machine Learning and Modeling

- `scikit-learn`
  - `StandardScaler` for feature scaling
  - `PCA` for linear dimensionality reduction
  - `KMeans` for subtype discovery
  - `AgglomerativeClustering` for comparison clustering
  - `RandomForestClassifier` for subtype prediction
  - cluster evaluation metrics such as silhouette score, Davies-Bouldin, and Calinski-Harabasz
- `umap-learn` for nonlinear 2D embedding
- `joblib` for model serialization

### Visualization

- `matplotlib`
- `seaborn`
- `plotly`

### App and Serving Layer

- `FastAPI` for the REST API
- `uvicorn` as the ASGI server
- `Streamlit` for the interactive dashboard
- `Jupyter` notebooks for exploratory and staged analysis

### Configuration and Utility

- `PyYAML` for config loading
- `httpx` and `pytest` for testing-oriented dependencies
- `openpyxl` for Excel export support in the dashboard

## Main Models Used

This project uses a mix of unsupervised and supervised models:

- `PCA`
  Reduces thousands of gene-level features into a smaller number of principal components.
- `UMAP`
  Builds a 2D representation for visualization while trying to preserve neighborhood structure.
- `K-Means`
  Main clustering model used to discover molecular subtypes.
- `Agglomerative Clustering`
  Secondary clustering approach used for comparison.
- `RandomForestClassifier`
  Supervised model trained on discovered subtype labels so new samples can be classified.

Saved model artifacts in `models/`:

- `scaler.pkl`
- `pca.pkl`
- `kmeans.pkl`
- `classifier.pkl`

## Configuration

Project settings live in [config.yaml](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/config.yaml:1).

Important configurable areas:

- input and output data paths,
- preprocessing thresholds,
- PCA component count,
- UMAP parameters,
- clustering search range for `K`,
- model save directory,
- results save directory.

Current major defaults:

- Remove bottom 20 percent of low-variance genes
- Keep top 5000 variable genes
- PCA to 50 components
- UMAP to 2 dimensions
- Test clusters from `K=2` to `K=9`

## Repository Structure

```text
Genome Data Clustering/
|-- api/
|   `-- app.py
|-- dashboard/
|   `-- app.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebook/
|   |-- 01_data_exploration.ipynb
|   |-- 02_preprocessing.ipynb
|   |-- 03_dimensionality_reduction.ipynb
|   |-- 04_clustering.ipynb
|   |-- 05_interpretation.ipynb
|   `-- 06_prediction_pipeline.ipynb
|-- results/
|-- src/
|   |-- preprocessor.py
|   |-- dimensionality.py
|   |-- clustering.py
|   |-- interpreter.py
|   |-- predictor.py
|   `-- run_pipeline.py
|-- environment.yml
|-- requirements.txt
`-- config.yaml
```

## Python Module Guide

### [src/preprocessor.py](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/src/preprocessor.py:1)

Purpose:
Prepares raw expression data for downstream machine learning.

What it does:

- loads config and resolves paths,
- reads raw expression and metadata files,
- separates embedded metadata rows from gene expression rows,
- transposes data into `cells x genes`,
- fills missing values,
- removes low-variance genes,
- keeps the most variable genes,
- applies `log1p` transformation,
- standardizes features with `StandardScaler`,
- saves processed expression and metadata outputs.

Key functions:

- `load_config`
- `resolve_path`
- `load_data`
- `separate_metadata`
- `filter_low_variance_genes`
- `keep_top_variable_genes`
- `handle_missing_values`
- `log_transform`
- `standardize`
- `run_preprocessing`

Why it matters:
This module reduces noise and ensures the clustering pipeline starts from a cleaner, more stable feature matrix.

### [src/dimensionality.py](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/src/dimensionality.py:1)

Purpose:
Compresses the feature space and creates a visual embedding.

What it does:

- runs PCA on scaled expression features,
- measures explained variance,
- plots per-component and cumulative PCA variance,
- runs UMAP on PCA outputs,
- creates a basic UMAP scatter plot,
- saves `X_pca.npy` and `X_umap.npy`.

Key functions:

- `run_pca`
- `plot_pca_variance`
- `run_umap`
- `plot_umap_basic`
- `run_dimensionality_reduction`

Why it matters:
Genomic data is extremely high-dimensional, so dimensionality reduction makes clustering faster, more stable, and easier to interpret visually.

### [src/clustering.py](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/src/clustering.py:1)

Purpose:
Finds subtype groups in the reduced feature space.

What it does:

- tests multiple `K` values for K-Means,
- evaluates cluster quality using inertia, silhouette, Davies-Bouldin, and Calinski-Harabasz scores,
- selects the best `K` by silhouette score,
- runs K-Means,
- runs hierarchical clustering for comparison,
- evaluates both clustering solutions,
- plots clusters on the UMAP embedding,
- saves cluster labels.

Key functions:

- `find_optimal_k`
- `plot_k_selection`
- `run_kmeans`
- `run_hierarchical`
- `evaluate_clusters`
- `plot_clusters_umap`
- `run_clustering`

Why it matters:
This is the subtype discovery stage of the project and the main unsupervised learning component.

### [src/interpreter.py](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/src/interpreter.py:1)

Purpose:
Helps explain what each cluster means biologically.

What it does:

- compares each cluster to all others gene by gene,
- runs Welch-style two-sample t-tests using `scipy.stats.ttest_ind(..., equal_var=False)`,
- computes fold-change style differences in mean expression,
- identifies the most significant genes per cluster,
- creates a cluster-sorted expression heatmap,
- creates top-gene significance bar plots.

Key functions:

- `find_top_genes_per_cluster`
- `plot_expression_heatmap`
- `plot_top_genes_barplot`

Why it matters:
Clustering alone gives labels, but this module helps turn those labels into interpretable molecular subtype stories.

### [src/predictor.py](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/src/predictor.py:1)

Purpose:
Converts the discovered unsupervised clusters into a deployable predictive model.

What it does:

- splits data into train and test sets,
- trains a `RandomForestClassifier`,
- prints a classification report,
- computes 5-fold cross-validation accuracy,
- generates a confusion matrix plot,
- saves all deployable model objects,
- predicts the subtype of a new input vector.

Key functions:

- `train_subtype_predictor`
- `save_all_models`
- `predict_new_patient`

Why it matters:
This module is what makes the project usable on future samples rather than only descriptive on the current dataset.

### [src/run_pipeline.py](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/src/run_pipeline.py:1)

Purpose:
Runs the full project pipeline from raw data to trained models and saved results.

What it does:

- calls preprocessing,
- calls dimensionality reduction,
- calls clustering,
- performs interpretation,
- trains the classifier,
- saves final artifacts and summary outputs.

Why it matters:
This is the main script to run when you want the entire system to execute end to end.

## API Layer

### [api/app.py](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/api/app.py:1)

Purpose:
Serves trained models through a FastAPI application.

Endpoints:

- `GET /`
  Returns service metadata and useful links.
- `GET /health`
  Returns a basic health response and whether the models directory exists.
- `POST /predict-subtype`
  Accepts gene expression values and returns predicted subtype, confidence, and class probabilities.
- `GET /cluster-summary`
  Returns a high-level subtype summary.

Technologies used:

- `FastAPI`
- `Pydantic`
- `joblib`
- `numpy`

Why it matters:
It turns the trained pipeline into a programmatic service that other applications can call.

## Dashboard Layer

### [dashboard/app.py](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/dashboard/app.py:1)

Purpose:
Provides a rich interactive interface for exploring cluster results and running prediction workflows.

What it includes:

- project overview metrics,
- subtype distribution visualizations,
- interactive UMAP and confidence views,
- cluster explorer,
- prediction upload workflow for new patient files,
- export tools for CSV and Excel outputs,
- image gallery for generated plots,
- cached loading of models, features, and results.

Technologies used:

- `Streamlit`
- `Plotly`
- `pandas`
- `numpy`
- `joblib`
- `openpyxl`

Why it matters:
It gives non-technical users a way to inspect analysis outputs and run predictions without using Python directly.

## Notebook Guide

The notebooks appear to mirror the pipeline in a phase-by-phase teaching and experimentation format.

### [notebook/01_data_exploration.ipynb](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/notebook/01_data_exploration.ipynb:1)

Purpose:
Initial inspection of the raw expression and metadata files.

What it covers:

- loading libraries,
- reading `config.yaml`,
- loading the raw expression matrix,
- checking matrix shape and sample values,
- loading metadata,
- verifying data types and structure.

Why it matters:
This notebook is the sanity-check entry point before preprocessing.

### [notebook/02_preprocessing.ipynb](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/notebook/02_preprocessing.ipynb:1)

Purpose:
Runs and verifies the preprocessing stage using `src/preprocessor.py`.

What it covers:

- executing `run_preprocessing`,
- inspecting outputs such as scaled data and metadata,
- generating verification plots,
- saving the scaler,
- saving processed arrays for later stages.

Why it matters:
This notebook makes the feature cleaning stage easier to inspect interactively.

### [notebook/03_dimensionality_reduction.ipynb](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/notebook/03_dimensionality_reduction.ipynb:1)

Purpose:
Runs PCA and UMAP using `src/dimensionality.py`.

What it covers:

- loading preprocessed data,
- running PCA,
- running UMAP,
- reviewing dimensionality reduction outputs,
- saving fitted reducers and embeddings.

Why it matters:
This notebook shows how the project transitions from thousands of genes to a compact representation suitable for clustering and visualization.

### [notebook/04_clustering.ipynb](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/notebook/04_clustering.ipynb:1)

Purpose:
Runs subtype discovery using `src/clustering.py`.

What it covers:

- loading the project config,
- searching for the best number of clusters,
- plotting selection metrics,
- running K-Means,
- running hierarchical clustering,
- evaluating cluster quality,
- plotting UMAP views of cluster assignments,
- saving outputs for later use.

Why it matters:
This notebook is where the main unsupervised subtype assignments are generated.

### [notebook/05_interpretation.ipynb](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/notebook/05_interpretation.ipynb:1)

Purpose:
Interprets the biological meaning of each cluster using `src/interpreter.py`.

What it covers:

- loading saved cluster labels,
- loading processed expression data,
- finding top genes per cluster,
- printing cluster-defining genes,
- producing expression heatmaps,
- producing subtype marker bar plots.

Why it matters:
This notebook connects subtype labels back to gene-level biological signals.

### [notebook/06_prediction_pipeline.ipynb](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/notebook/06_prediction_pipeline.ipynb:1)

Purpose:
Builds the deployable prediction layer using `src/predictor.py`.

What it covers:

- loading outputs from previous stages,
- training the `RandomForestClassifier`,
- generating evaluation metrics,
- saving all model artifacts,
- testing a sample prediction workflow,
- preparing the project for API deployment.

Why it matters:
This notebook is the bridge from clustering research to a usable predictive system.

## Outputs Produced by the Pipeline

The repo already includes several result artifacts, and the pipeline is designed to generate files like these:

- `results/04_pca_variance.png`
- `results/05_umap_basic.png`
- `results/06_k_selection.png`
- `results/07_kmeans_clusters.png`
- `results/08_hierarchical_clusters.png`
- `results/09_expression_heatmap.png`
- `results/10_top_genes.png`
- `results/11_confusion_matrix.png`
- `results/cluster_assignments.csv`
- `results/top_genes_per_cluster.csv`
- `results/prediction_results.csv`
- `results/X_pca.npy`
- `results/X_umap.npy`

## How to Run the Project

### Option 1: Run the Full Python Pipeline

```powershell
python src/run_pipeline.py
```

### Option 2: Run the API

```powershell
uvicorn api.app:app --reload
```

Open:

- `http://localhost:8000`
- `http://localhost:8000/docs`

### Option 3: Run the Streamlit Dashboard

```powershell
streamlit run dashboard/app.py
```

Open:

- `http://localhost:8501`

### Option 4: Use the Notebooks

Open Jupyter and run notebooks in order:

1. `01_data_exploration.ipynb`
2. `02_preprocessing.ipynb`
3. `03_dimensionality_reduction.ipynb`
4. `04_clustering.ipynb`
5. `05_interpretation.ipynb`
6. `06_prediction_pipeline.ipynb`

## Environment Files

### [requirements.txt](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/requirements.txt:1)

This is the pip-focused dependency list for lightweight Python environments and reproducible installs.

### [environment.yml](/abs/path/c:/Users/DELL/OneDrive/Documents/Machine%20Learning/ML_Projects/Unsupervised%20Learning/Genome%20Data%20Clustering/environment.yml:1)

This is the Conda environment definition for local development and notebook use.

## Strengths of the Current Project

- Clean separation between data science modules and deployment modules
- Reusable pipeline functions rather than notebook-only logic
- Multiple interfaces: notebooks, API, and dashboard
- Good use of dimensionality reduction before clustering
- Clear progression from unsupervised discovery to supervised prediction
- Saved artifacts make deployment practical

## Current Assumptions and Notes

- The preprocessing code assumes the raw expression file contains metadata rows in the first three rows.
- The prediction API expects gene expression inputs aligned with the feature set used during training.
- The classifier is trained on cluster labels rather than external clinical ground-truth subtype labels.
- Some output files shown by the dashboard are expected to exist after a successful pipeline run.

## Suggested Future Improvements

- Add formal tests for pipeline stages and API routes.
- Add data validation checks for feature length and schema consistency.
- Add model versioning and experiment tracking.
- Add clearer biological annotation mapping from cluster ID to named subtype.
- Add benchmark comparisons across more clustering and classification models.
- Add reproducible evaluation reports for different parameter settings.

## Summary

This repository is a full-stack applied machine learning project for genomic subtype discovery and prediction. It combines classical preprocessing, PCA, UMAP, clustering, differential expression analysis, and Random Forest classification with deployable interfaces in FastAPI, Streamlit, and Jupyter. It is both an analysis workflow and a prototype deployment system for gene-expression-based subtype prediction.
>>>>>>> 171d50d (Initial portfolio-ready version)
